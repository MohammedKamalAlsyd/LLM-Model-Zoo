import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    MinPLogitsWarper,
)


# ==========================================
# 1. Configurations & Constants
# ==========================================
LLAMA_520M_CONFIG: Dict[str, Any] = {
    "vocab_size": 8,
    "max_position_embeddings": 131072,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "attn_implementation": "sdpa",
    "head_dim": 64,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "model_type": "llama",
    "num_key_value_heads": 16,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
}

@dataclass
class T3Config:
    text_tokens_dict_size: int = 704      # 704 (EN) or 2454 (Multilingual)
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_tokens: int = 2048

    speech_tokens_dict_size: int = 8194
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    max_speech_tokens: int = 4096

    llama_config_name: str = "Llama_520M"
    input_pos_emb: str = "learned"
    speaker_embed_size: int = 256
    emotion_adv: bool = True
    
    @property
    def n_channels(self) -> int:
        return 1024


@dataclass
class T3Cond:
    """Dataclass aligning inputs from `voice_encoder` and `S3Tokenizer` to T3."""
    speaker_emb: torch.Tensor                                     # From VoiceEncoder [1, 256]
    cond_prompt_speech_emb: Optional[torch.Tensor] = None        # From S3Tokenizer (Embedded)
    emotion_adv: Optional[Union[torch.Tensor, float]] = 0.5      # Accepts scalar float or Tensor


# ==========================================
# 2. Sub-Modules (Pos Embeds & Perceiver)
# ==========================================
class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(torch.arange(x.shape[1], device=x.device))

    def get_fixed_embedding(self, idx: Union[int, torch.Tensor]) -> torch.Tensor:
        device = self.emb.weight.device
        idx_tensor = idx if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        idx_tensor = torch.atleast_2d(idx_tensor).to(device)
        return self.emb(idx_tensor)


class AttentionBlock(nn.Module):
    """Condensed Perceiver Attention Block preserving exact weight signatures."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1_n, x2_n = self.norm(x1), self.norm(x2)
        
        # Split heads
        q = self.to_q(x1_n).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        k = self.to_k(x2_n).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v = self.to_v(x2_n).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        
        # Scaled Dot-Product Attention (without the relative bias)
        sim = torch.einsum("bhlt,bhst->bhls", q, k) * (q.size(-1) ** -0.5)
        out = torch.einsum("bhls,bhst->bhlt", F.softmax(sim, dim=-1), v)
        
        # Combine heads
        out = out.transpose(1, 2).flatten(-2)
        return x1 + self.proj_out(out)


class Perceiver(nn.Module):
    """Compresses variable length audio prompts down to exactly 32 tokens."""
    def __init__(self, query_tokens: int = 32, dim: int = 1024, heads: int = 4):
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, query_tokens, dim))
        self.pre_attention_query.data.uniform_(-0.05, 0.05)
        self.attn = AttentionBlock(dim, heads)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.pre_attention_query.expand(h.size(0), -1, -1)
        pre_att = self.attn(q, h)          # Cross-Attention
        return self.attn(pre_att, pre_att) # Self-Attention


class T3CondEnc(nn.Module):
    def __init__(self, hp: T3Config):
        super().__init__()
        self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False) if hp.emotion_adv else None
        self.perceiver = Perceiver(dim=hp.n_channels)

    def forward(self, cond: T3Cond) -> torch.Tensor:
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, 256))[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])
        
        cond_prompt = self.perceiver(cond.cond_prompt_speech_emb) if cond.cond_prompt_speech_emb is not None else empty
        
        # Safely convert float or tensor to proper shape for emotion embedding
        if self.emotion_adv_fc is not None and cond.emotion_adv is not None:
            if torch.is_tensor(cond.emotion_adv):
                emo_tensor = cond.emotion_adv.to(cond_spkr.device, dtype=cond_spkr.dtype)
            else:
                emo_tensor = torch.tensor(cond.emotion_adv, device=cond_spkr.device, dtype=cond_spkr.dtype)
            cond_emo = self.emotion_adv_fc(emo_tensor.view(-1, 1, 1))
        else:
            cond_emo = empty
        
        # Concat Shape: [B, 1(Spkr) + 32(Prompt) + 1(Emo), 1024]
        return torch.cat((cond_spkr, empty, cond_prompt, cond_emo), dim=1)


# ==========================================
# 3. Main T3 Model
# ==========================================
class T3(nn.Module):
    """Chatterbox T3 Decoder-Only TTS SubModel."""
    def __init__(self, hp: T3Config = T3Config()):
        super().__init__()
        self.hp = hp
        self.dim = hp.n_channels

        # Using from_dict avoids Pylance **dict unpacking type collisions
        self.cfg = LlamaConfig.from_dict(LLAMA_520M_CONFIG)
        self.tfmr = LlamaModel(self.cfg)

        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        self.text_pos_emb = LearnedPositionEmbeddings(hp.max_text_tokens + 2, self.dim)
        self.speech_pos_emb = LearnedPositionEmbeddings(hp.max_speech_tokens + 4, self.dim)

        self.text_head = nn.Linear(self.dim, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.dim, hp.speech_tokens_dict_size, bias=False)

    @property
    def device(self) -> torch.device:
        return self.speech_head.weight.device

    def prepare_input_embeds(
        self, t3_cond: T3Cond, text_tokens: torch.Tensor, speech_tokens: torch.Tensor, cfg_weight: float = 0.0
    ):
        cond_emb = self.cond_enc(t3_cond)  
        
        text_emb = self.text_emb(text_tokens) + self.text_pos_emb(text_tokens)
        speech_emb = self.speech_emb(speech_tokens) + self.speech_pos_emb(speech_tokens)

        # Unconditional pass for CFG (Zeroes out text tokens for batch index 1)
        if cfg_weight > 0.0 and text_emb.size(0) > 1:
            text_emb[1].zero_()

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        embeds = torch.cat([cond_emb, text_emb, speech_emb], dim=1)
        return embeds, cond_emb.size(1)

    def forward(
        self, 
        t3_cond: T3Cond, 
        text_tokens: torch.Tensor, 
        text_lens: torch.Tensor, 
        speech_tokens: torch.Tensor, 
        speech_lens: torch.Tensor
    ):
        embeds, len_cond = self.prepare_input_embeds(t3_cond, text_tokens, speech_tokens)
        hidden_states = self.tfmr(inputs_embeds=embeds, use_cache=False).last_hidden_state

        B, _, dim = hidden_states.shape
        text_latents = torch.zeros(B, text_tokens.size(1), dim, device=self.device)
        speech_latents = torch.zeros(B, speech_tokens.size(1), dim, device=self.device)
        
        for i in range(B):
            t_end = len_cond + text_lens[i].item()
            s_start = len_cond + text_tokens.size(1)
            s_end = s_start + speech_lens[i].item()
            text_latents[i, :text_lens[i]] = hidden_states[i, len_cond:t_end]
            speech_latents[i, :speech_lens[i]] = hidden_states[i, s_start:s_end]

        return self.text_head(text_latents), self.speech_head(speech_latents)

    def loss(
        self, 
        t3_cond: T3Cond, 
        text_tokens: torch.Tensor, 
        text_lens: torch.Tensor, 
        speech_tokens: torch.Tensor, 
        speech_lens: torch.Tensor
    ):
        t_logits, s_logits = self.forward(t3_cond, text_tokens, text_lens, speech_tokens, speech_lens)
        
        mask_t = torch.arange(text_tokens.size(1), device=self.device)[None] >= text_lens[:, None]
        mask_s = torch.arange(speech_tokens.size(1), device=self.device)[None] >= speech_lens[:, None]
        
        loss_t = F.cross_entropy(t_logits.transpose(1, 2), text_tokens.masked_fill(mask_t, -100), ignore_index=-100)
        loss_s = F.cross_entropy(s_logits, speech_tokens.masked_fill(mask_s, -100), ignore_index=-100)
        return loss_t, loss_s

    @torch.inference_mode()
    def generate(
        self, 
        t3_cond: T3Cond, 
        text_tokens: torch.Tensor, 
        max_new_tokens: int = 1000,
        temperature: float = 0.8, 
        top_p: float = 0.95, 
        min_p: float = 0.05, 
        repetition_penalty: float = 1.2, 
        cfg_weight: float = 3.0
    ) -> torch.Tensor:
        """Optimized custom loop with built-in Classifier-Free Guidance (CFG)."""
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)
        
        # 1. Expand batch for CFG [Conditional, Unconditional]
        if cfg_weight > 0.0:
            text_tokens = text_tokens.repeat(2, 1)
            if t3_cond.cond_prompt_speech_emb is not None:
                t3_cond.cond_prompt_speech_emb = t3_cond.cond_prompt_speech_emb.repeat(2, 1, 1)
            t3_cond.speaker_emb = t3_cond.speaker_emb.repeat(2, 1)
            if t3_cond.emotion_adv is not None:
                val = float(t3_cond.emotion_adv) if not torch.is_tensor(t3_cond.emotion_adv) else t3_cond.emotion_adv.item()
                t3_cond.emotion_adv = torch.tensor([val, val], device=self.device)
        
        # 2. Get initial Embeds
        bos = torch.tensor([[self.hp.start_speech_token]], device=self.device).repeat(text_tokens.size(0), 1)
        inputs_embeds, _ = self.prepare_input_embeds(t3_cond, text_tokens, bos, cfg_weight=cfg_weight)

        # Logits Processors
        rep_pen = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        min_p_warp = MinPLogitsWarper(min_p=min_p)
        top_p_warp = TopPLogitsWarper(top_p=top_p)

        generated_ids = bos[:1].clone()
        predicted = []
        
        # Forward pass 1: Full Context
        out = self.tfmr(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        
        # Forward pass N: Token by Token
        for i in range(max_new_tokens):
            hidden = out.last_hidden_state[:, -1, :]
            logits = self.speech_head(hidden)

            # Extract CFG
            if cfg_weight > 0.0:
                cond, uncond = logits[0:1], logits[1:2]
                logits = cond + cfg_weight * (cond - uncond)
            
            # Processing & Sampling
            logits = rep_pen(generated_ids, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = min_p_warp(generated_ids, logits)
            logits = top_p_warp(generated_ids, logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.hp.stop_speech_token:
                break

            # Embed next token for next step
            next_embed = self.speech_emb(next_token) + self.speech_pos_emb.get_fixed_embedding(i + 1)
            if cfg_weight > 0.0:
                next_embed = next_embed.repeat(2, 1, 1)
            
            out = self.tfmr(inputs_embeds=next_embed, past_key_values=past, use_cache=True)
            past = out.past_key_values

        return torch.cat(predicted, dim=1)