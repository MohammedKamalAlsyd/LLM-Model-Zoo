"""Chatterbox T3: 520M-Parameter Autoregressive Transformer with Perceiver Resampler.

Predicts discrete speech tokens conditioned on text tokens, speaker d-vectors,
emotion scalar embeddings, and reference acoustic prompts with Classifier-Free Guidance (CFG).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from Zoo.Chatterbox.configs import T3Config


@dataclass
class T3Cond:
    """Multimodal conditioning context for T3 speech generation."""
    speaker_emb: torch.Tensor                                     # (batch_size, 256) VoiceEncoder d-vector
    clap_emb: Optional[torch.Tensor] = None                       # Reserved for acoustic CLAP parity
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None      # (batch_size, T_prompt) S3 discrete codes
    cond_prompt_speech_emb: Optional[torch.Tensor] = None         # (batch_size, T_prompt, 1024) Embedded internally
    emotion_adv: Optional[Union[torch.Tensor, float]] = 0.5       # Scalar emotion intensity (0.0 to 1.0)

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "T3Cond":
        """Transfers tensor attributes to the specified device and precision."""
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_floating = v.dtype in (torch.float16, torch.float32, torch.bfloat16)
                setattr(self, k, v.to(device=device, dtype=dtype if is_floating else None))
        return self


class LearnedPositionEmbeddings(nn.Module):
    """Learned 1D absolute position embeddings with normal initialization."""

    def __init__(self, seq_len: int, model_dim: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Input tensor of shape (batch_size, seq_len, ...).

        Returns:
            Position embedding tensor of shape (seq_len, model_dim).
        """
        return self.emb(torch.arange(x.shape[1], device=x.device))

    def get_fixed_embedding(self, idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """Extracts position vector for single-step autoregressive decode increments."""
        device = self.emb.weight.device
        idx_tensor = idx if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        return self.emb(torch.atleast_2d(idx_tensor).to(device))


class AttentionBlock2(nn.Module):
    """Cross/Self-Attention Block with Pre-LayerNorm for the Perceiver Resampler."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Args:

            x1: Query source tensor of shape (batch_size, len_q, channels).
            x2: Key/Value source tensor of shape (batch_size, len_kv, channels).

        Returns:
            Attended residual output of shape (batch_size, len_q, channels).
        """
        b, l1, c = x1.shape
        _, l2, _ = x2.shape

        q = self.to_q(self.norm(x1)).view(b, l1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(self.norm(x2)).view(b, l2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(self.norm(x2)).view(b, l2, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, l1, c)
        return x1 + self.proj_out(out)


class Perceiver(nn.Module):
    """Compresses variable-length speech prompt embeddings into fixed-length latent tokens."""

    def __init__(
        self,
        pre_attention_query_token: int = 32,
        embedding_dim: int = 1024,
        num_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, pre_attention_query_token, embedding_dim))
        self.pre_attention_query.data.uniform_(-0.05, 0.05)
        self.attn = AttentionBlock2(channels=embedding_dim, num_heads=num_attn_heads)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Args:

            h: Uncompressed speech prompt embeddings of shape (batch_size, seq_len, 1024).

        Returns:
            Compressed prompt tokens of shape (batch_size, 32, 1024).
        """
        q = self.pre_attention_query.expand(h.size(0), -1, -1)
        cross_attended = self.attn(q, h)
        return self.attn(cross_attended, cross_attended)


class T3CondEnc(nn.Module):
    """Projects and combines heterogeneous prompt conditions into a unified prefix sequence."""

    def __init__(self, cfg: T3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.spkr_enc = nn.Linear(cfg.speaker_embed_size, cfg.n_channels)
        self.emotion_adv_fc = nn.Linear(1, cfg.n_channels, bias=False) if cfg.emotion_adv else None
        self.perceiver = Perceiver(
            pre_attention_query_token=cfg.perceiver_query_tokens,
            embedding_dim=cfg.n_channels,
            num_attn_heads=cfg.perceiver_num_heads,
        )

    def forward(self, cond: T3Cond) -> torch.Tensor:
        """Args:

            cond: T3Cond object holding speaker d-vector, prompt tokens, and emotion value.

        Returns:
            Unified prefix tensor of shape (batch_size, 34, 1024):
            1 speaker token + 32 prompt tokens + 1 emotion token.
        """
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.cfg.speaker_embed_size))[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])

        cond_prompt = (
            self.perceiver(cond.cond_prompt_speech_emb)
            if cond.cond_prompt_speech_emb is not None
            else empty
        )

        cond_emo = empty
        if self.emotion_adv_fc is not None and cond.emotion_adv is not None:
            emo_val = (
                cond.emotion_adv
                if torch.is_tensor(cond.emotion_adv)
                else torch.tensor(cond.emotion_adv)
            )
            cond_emo = self.emotion_adv_fc(
                emo_val.to(device=cond_spkr.device, dtype=cond_spkr.dtype).view(-1, 1, 1)
            )

        return torch.cat((cond_spkr, empty, cond_prompt, cond_emo), dim=1)


class T3(nn.Module):
    """Chatterbox T3 Autoregressive Multilingual Backbone."""

    def __init__(self, config: Optional[T3Config] = None) -> None:
        super().__init__()
        self.hp = config or T3Config()
        self.dim = self.hp.n_channels

        llama_dict: Dict[str, Any] = {
            "vocab_size": 8,
            "max_position_embeddings": self.hp.max_position_embeddings,
            "hidden_size": self.hp.hidden_size,
            "intermediate_size": self.hp.intermediate_size,
            "num_hidden_layers": self.hp.num_hidden_layers,
            "num_attention_heads": self.hp.num_attention_heads,
            "attn_implementation": "sdpa",
            "head_dim": self.hp.head_dim,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "model_type": "llama",
            "num_key_value_heads": self.hp.num_key_value_heads,
            "rms_norm_eps": self.hp.rms_norm_eps,
            "rope_theta": self.hp.rope_theta,
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
        self.llama_config = LlamaConfig.from_dict(llama_dict)
        self.tfmr = LlamaModel(self.llama_config)

        self.cond_enc = T3CondEnc(self.hp)
        self.text_emb = nn.Embedding(self.hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.hp.speech_tokens_dict_size, self.dim)

        self.text_pos_emb = LearnedPositionEmbeddings(self.hp.max_text_tokens + 2, self.dim)
        self.speech_pos_emb = LearnedPositionEmbeddings(self.hp.max_speech_tokens + 4, self.dim)

        self.speech_head = nn.Linear(self.dim, self.hp.speech_tokens_dict_size, bias=False)

    @property
    def device(self) -> torch.device:
        return self.speech_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.speech_head.weight.dtype

    def prepare_conditioning(self, t3_cond: T3Cond) -> torch.Tensor:
        """Embeds discrete speech tokens and executes Perceiver prompt compression."""
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = (
                self.speech_emb(t3_cond.cond_prompt_speech_tokens)
                + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            )
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        cfg_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, int]:
        """Constructs concatenated sequence: [Conditioning (34) + Text (N) + Speech (M)]."""
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens) + self.text_pos_emb(text_tokens)
        speech_emb = self.speech_emb(speech_tokens) + self.speech_pos_emb(speech_tokens)

        # Unconditional zeroing on batch slice 1 for Classifier-Free Guidance
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
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Autoregressive speech token inference with CFG combining and KV caching.

        Args:
            t3_cond: Multimodal conditioning container.
            text_tokens: LongTensor of shape (1, seq_len) with [START] and [STOP] tags.
            max_new_tokens: Maximum acoustic tokens to sample (25 tokens/second).
            temperature: Softmax temperature scale.
            top_p: Top-P nucleus sampling threshold.
            min_p: Min-P sampling probability floor.
            repetition_penalty: Logit penalty applied to previously generated tokens.
            cfg_weight: Guidance scale ($s$). Logits calculated as:
                        $$\text{logits} = \mathbf{l}_{\text{cond}} + s \cdot (\mathbf{l}_{\text{cond}} - \mathbf{l}_{\text{uncond}})$$

        Returns:
            LongTensor of predicted speech tokens of shape (1, num_generated_tokens).
        """
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)

        # Expand batch to 2 for concurrent conditional and unconditional evaluation
        if cfg_weight > 0.0 and text_tokens.size(0) == 1:
            text_tokens = text_tokens.repeat(2, 1)

        b = text_tokens.size(0)
        bos_tokens = torch.full((b, 1), self.hp.start_speech_token, dtype=torch.long, device=self.device)
        inputs_embeds, _ = self.prepare_input_embeds(t3_cond, text_tokens, bos_tokens, cfg_weight=cfg_weight)

        rep_pen_proc = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        min_p_warp = MinPLogitsWarper(min_p=float(min_p))
        top_p_warp = TopPLogitsWarper(top_p=float(top_p))

        generated_ids = bos_tokens[:1].clone()
        predicted_tokens: List[torch.Tensor] = []

        # Prefill pass over full prompt prefix
        out = self.tfmr(inputs_embeds=inputs_embeds, use_cache=True)
        past_key_values = out.past_key_values

        for i in range(max_new_tokens):
            last_hidden = out.last_hidden_state[:, -1:, :]
            logits = self.speech_head(last_hidden).squeeze(1)

            # CFG Logit Extrapolation
            if cfg_weight > 0.0 and b > 1:
                cond_l, uncond_l = logits[0:1], logits[1:2]
                logits = cond_l + cfg_weight * (cond_l - uncond_l)
            else:
                logits = logits[0:1]

            # Sampling Warpers
            ids_for_pen = generated_ids[:1]
            logits = rep_pen_proc(ids_for_pen, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = min_p_warp(ids_for_pen, logits)
            logits = top_p_warp(ids_for_pen, logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            predicted_tokens.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.hp.stop_speech_token:
                break

            # Incremental single-token step using cached KV
            next_embed = self.speech_emb(next_token) + self.speech_pos_emb.get_fixed_embedding(i + 1)
            if b > 1:
                next_embed = next_embed.expand(b, -1, -1)

            out = self.tfmr(inputs_embeds=next_embed, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values

        if not predicted_tokens:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)

        return torch.cat(predicted_tokens, dim=1)