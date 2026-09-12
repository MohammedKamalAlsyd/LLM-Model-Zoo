import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from ..utils.KVCache import KVCache
from ..utils.rotate_functions import TextMRotaryEmbedding, apply_rotary_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        return (x_f * torch.rsqrt(variance + self.eps)).to(orig_dtype) * self.weight


class TextMLP(nn.Module):
    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9728):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2560,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        layer_idx: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Per-head RMSNorm (Qwen3 specific: applied to the head_dim)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape

        q = self.q_proj(x).view(bs, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bs, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bs, seq_len, self.num_kv_heads, self.head_dim)

        # Head normalization
        q = self.q_norm(q).transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (bs, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)

        # Apply 3D M-RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Cache KV
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # Repeat KV for Grouped Query Attention
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        is_causal = mask is None and seq_len > 1
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.o_proj(out)


class TextDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 2560, num_heads: int = 32, num_kv_heads: int = 8, head_dim: int = 128, intermediate_size: int = 9728, layer_idx: int = 0, eps: float = 1e-6):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = TextAttention(hidden_size, num_heads, num_kv_heads, head_dim, layer_idx, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = TextMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos=cos, sin=sin, kv_cache=kv_cache, mask=mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3VLTextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_theta: float = 5000000.0,
        mrope_section: Sequence[int] = (24, 20, 20),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = TextMRotaryEmbedding(head_dim=head_dim, theta=rope_theta, mrope_section=mrope_section)
        self.layers = nn.ModuleList([
            TextDecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, layer_idx=i, eps=eps)
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: (bs, seq_len, hidden_size)
            position_ids: (3, bs, seq_len) 3D coordinates [T, H, W]
            visual_pos_mask: (bs, seq_len) boolean tensor indicating visual token positions
            deepstack_embeds: list of (num_visual_tokens, hidden_size) tensors
        """
        cos, sin = self.rotary_emb(position_ids)
        h = inputs_embeds

        for i, layer in enumerate(self.layers):
            h = layer(h, cos=cos, sin=sin, kv_cache=kv_cache, mask=mask)

            # DeepStack injection into early transformer layers
            if deepstack_embeds is not None and i < len(deepstack_embeds) and visual_pos_mask is not None:
                h = h.clone()
                h[visual_pos_mask] = h[visual_pos_mask] + deepstack_embeds[i].to(h.dtype)

        return self.norm(h)