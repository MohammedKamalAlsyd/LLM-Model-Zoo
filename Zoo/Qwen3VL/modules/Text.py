"""Qwen3-VL Autoregressive Language Model with 3D M-RoPE, Per-Head RMSNorm, and DeepStack."""

from collections.abc import Sequence
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Common.attention_utils import repeat_kv
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.RoPE import apply_rotary_pos_emb
from Zoo.Qwen3VL.configs import Qwen3VLTextConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization evaluated in FP32 precision."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        normed = x_f * torch.rsqrt(variance + self.eps)
        return (normed * self.weight).to(orig_dtype)


class TextMRotaryEmbedding(nn.Module):
    """3D Multimodal Rotary Position Embedding (M-RoPE).

    Decomposes position frequencies across three spatial-temporal axes:
    Temporal (T), Height (H), and Width (W) using section partitioning.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int = 128,
        theta: float = 5000000.0,
        mrope_section: Sequence[int] = (24, 20, 20),
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.mrope_section = list(mrope_section)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates (cos, sin) frequency tensors for 3D coordinates.

        Args:
            position_ids: Tensor of shape (3, batch_size, seq_len) with [T, H, W] positions.

        Returns:
            cos, sin: Frequency tensors of shape (batch_size, seq_len, head_dim)
                     broadcastable over attention heads.
        """
        # Outer product: (3, bs, seq_len, 1) * (1, 1, 1, head_dim // 2)
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq[None, None, None, :]

        freqs_t = freqs[0]  # Base Temporal channel
        for dim, offset in enumerate((1, 2), start=1):  # 1=Height, 2=Width
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        freqs_thw = torch.cat((freqs_t, freqs_t), dim=-1)
        return freqs_thw.cos(), freqs_thw.sin()


class TextMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9728) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextAttention(nn.Module):
    """Grouped-Query Attention with Per-Head RMSNorm and 3D M-RoPE."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Qwen3 architectural hallmark: Per-head RMSNorm before attention projection
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape

        q = self.q_proj(x).view(bs, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bs, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bs, seq_len, self.num_kv_heads, self.head_dim)

        # Apply per-head normalization prior to RoPE rotation
        q = self.q_norm(q).transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (bs, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)

        # Apply 3D M-RoPE using unified Zoo.Common.RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Autoregressive key/value caching
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # Expand Key/Value heads for Grouped-Query Attention
        k = repeat_kv(k, self.num_groups)
        v = repeat_kv(v, self.num_groups)

        is_causal = (mask is None and seq_len > 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.o_proj(out)


class TextDecoderLayer(nn.Module):
    """Standard Pre-Norm Transformer Decoder Block."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TextAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TextMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos=cos, sin=sin, kv_cache=kv_cache, mask=mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3VLTextModel(nn.Module):
    """Qwen3-VL Language Model Backbone with DeepStack feature injection."""

    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = TextMRotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        self.layers = nn.ModuleList([
            TextDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with DeepStack intermediate representation injection.

        Args:
            inputs_embeds: Fused embedding sequence of shape (bs, seq_len, hidden_size).
            position_ids: 3D coordinates tensor of shape (3, bs, seq_len).
            kv_cache: Key-value cache instance for incremental decoding.
            mask: Optional attention mask tensor.
            visual_pos_mask: Boolean tensor (bs, seq_len) locating visual token positions.
            deepstack_embeds: List of intermediate features injected additively into early layers.

        Returns:
            Normalized last hidden states of shape (bs, seq_len, hidden_size).
        """
        cos, sin = self.rotary_emb(position_ids)
        h = inputs_embeds

        for i, layer in enumerate(self.layers):
            h = layer(h, cos=cos, sin=sin, kv_cache=kv_cache, mask=mask)

            # DeepStack injection: Add intermediate vision representations directly into early layers
            if deepstack_embeds is not None and i < len(deepstack_embeds) and visual_pos_mask is not None:
                h = h.clone()
                h[visual_pos_mask] = h[visual_pos_mask] + deepstack_embeds[i].to(h.dtype)

        return self.norm(h)