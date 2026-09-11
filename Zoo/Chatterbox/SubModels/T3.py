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
    text_tokens_dict_size: int = 2454     # 2454 for Multilingual v3 (704 for EN-only)
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
    speech_cond_prompt_len: int = 150
    emotion_adv: bool = True

    @property
    def n_channels(self) -> int:
        return 1024

    @classmethod
    def multilingual(cls):
        return cls(text_tokens_dict_size=2454)

    @classmethod
    def english_only(cls):
        return cls(text_tokens_dict_size=704)


@dataclass
class T3Cond:
    speaker_emb: torch.Tensor                                     # [1, 256] from VoiceEncoder
    clap_emb: Optional[torch.Tensor] = None                       # Unused in v3, kept for signature
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None     # [1, T] from S3Tokenizer
    cond_prompt_speech_emb: Optional[torch.Tensor] = None        # Embedded internally by T3
    emotion_adv: Optional[Union[torch.Tensor, float]] = 0.5

    def to(self, device=None, dtype=None):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = v.dtype in (torch.float16, torch.float32, torch.bfloat16)
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self


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
        return self.emb(torch.atleast_2d(idx_tensor).to(device))


class AttentionBlock2(nn.Module):
    """Matches the exact naming and structure of Chatterbox's Perceiver AttentionBlock2."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, L1, C = x1.shape
        _, L2, _ = x2.shape

        q = self.to_q(self.norm(x1)).view(B, L1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(self.norm(x2)).view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(self.norm(x2)).view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, L1, C)
        return x1 + self.proj_out(out)


class Perceiver(nn.Module):
    def __init__(self, pre_attention_query_token: int = 32, embedding_dim: int = 1024, num_attn_heads: int = 4):
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, pre_attention_query_token, embedding_dim))
        self.pre_attention_query.data.uniform_(-0.05, 0.05)
        self.attn = AttentionBlock2(embedding_dim, num_attn_heads)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.pre_attention_query.expand(h.size(0), -1, -1)
        pre_att = self.attn(q, h)          # Cross-attention
        return self.attn(pre_att, pre_att) # Self-attention


class T3CondEnc(nn.Module):
    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp
        self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False) if hp.emotion_adv else None
        self.perceiver = Perceiver(embedding_dim=hp.n_channels)

    def forward(self, cond: T3Cond) -> torch.Tensor:
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])

        cond_prompt = self.perceiver(cond.cond_prompt_speech_emb) if cond.cond_prompt_speech_emb is not None else empty

        cond_emo = empty
        if self.emotion_adv_fc is not None and cond.emotion_adv is not None:
            emo_val = cond.emotion_adv if torch.is_tensor(cond.emotion_adv) else torch.tensor(cond.emotion_adv)
            cond_emo = self.emotion_adv_fc(emo_val.to(device=cond_spkr.device, dtype=cond_spkr.dtype).view(-1, 1, 1))

        # Shape: [B, 1 (spkr) + 32 (prompt) + 1 (emo), 1024]
        return torch.cat((cond_spkr, empty, cond_prompt, cond_emo), dim=1)


# ==========================================
# 3. Main T3 Model
# ==========================================
class T3(nn.Module):
    """Chatterbox T3 Multilingual Autoregressive Backbone."""
    def __init__(self, hp: Optional[T3Config] = None):
        super().__init__()
        self.hp = hp or T3Config.multilingual()
        self.dim = self.hp.n_channels

        self.cfg = LlamaConfig.from_dict(LLAMA_520M_CONFIG)
        self.tfmr = LlamaModel(self.cfg)

        self.cond_enc = T3CondEnc(self.hp)
        self.text_emb = nn.Embedding(self.hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.hp.speech_tokens_dict_size, self.dim)

        self.text_pos_emb = LearnedPositionEmbeddings(self.hp.max_text_tokens + 2, self.dim)
        self.speech_pos_emb = LearnedPositionEmbeddings(self.hp.max_speech_tokens + 4, self.dim)

        self.speech_head = nn.Linear(self.dim, self.hp.speech_tokens_dict_size, bias=False)

    @property
    def device(self) -> torch.device:
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        # Embed discrete prompt tokens using speech and position embeddings
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = (
                self.speech_emb(t3_cond.cond_prompt_speech_tokens)
                + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            )
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self, t3_cond: T3Cond, text_tokens: torch.Tensor, speech_tokens: torch.Tensor, cfg_weight: float = 0.0
    ):
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens) + self.text_pos_emb(text_tokens)
        speech_emb = self.speech_emb(speech_tokens) + self.speech_pos_emb(speech_tokens)

        # Apply Unconditional masking for CFG (batch index 1 is unconditional)
        if cfg_weight > 0.0 and text_emb.size(0) > 1:
            text_emb[1].zero_()

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        embeds = torch.cat([cond_emb, text_emb, speech_emb], dim=1)
        return embeds, cond_emb.size(1)

    @torch.inference_mode()
    def inference(
        self,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> torch.Tensor:
        """Generation method called directly by ChatterboxMultilingualTTS."""
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)

        # If batch size is 1 and CFG is requested, expand to batch size 2 [cond, uncond]
        if cfg_weight > 0.0 and text_tokens.size(0) == 1:
            text_tokens = text_tokens.repeat(2, 1)

        B = text_tokens.size(0)
        bos_tokens = torch.full((B, 1), self.hp.start_speech_token, dtype=torch.long, device=self.device)
        inputs_embeds, _ = self.prepare_input_embeds(t3_cond, text_tokens, bos_tokens, cfg_weight=cfg_weight)

        # Instantiate logit processors
        rep_pen_proc = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        min_p_warp = MinPLogitsWarper(min_p=float(min_p))
        top_p_warp = TopPLogitsWarper(top_p=float(top_p))

        generated_ids = bos_tokens[:1].clone()
        predicted_tokens = []

        # Forward pass 1: Process the full prefix context
        out = self.tfmr(inputs_embeds=inputs_embeds, use_cache=True)
        past_key_values = out.past_key_values

        for i in range(max_new_tokens):
            last_hidden = out.last_hidden_state[:, -1:, :]
            logits = self.speech_head(last_hidden).squeeze(1)  # (B, Vocab)

            # CFG combining
            if cfg_weight > 0.0 and B > 1:
                cond, uncond = logits[0:1], logits[1:2]
                logits = cond + cfg_weight * (cond - uncond)
            else:
                logits = logits[0:1]

            # Sample next token
            ids_for_pen = generated_ids[:1]
            logits = rep_pen_proc(ids_for_pen, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = min_p_warp(ids_for_pen, logits)
            logits = top_p_warp(ids_for_pen, logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            predicted_tokens.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.hp.stop_speech_token:
                break

            # Forward pass N: Single token step with cached KV
            next_embed = self.speech_emb(next_token) + self.speech_pos_emb.get_fixed_embedding(i + 1)
            if B > 1:
                next_embed = next_embed.expand(B, -1, -1)

            out = self.tfmr(inputs_embeds=next_embed, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values

        return torch.cat(predicted_tokens, dim=1)