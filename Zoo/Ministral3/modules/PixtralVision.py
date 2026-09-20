"""Compact, High-Performance Pixtral Vision Encoder with SDPA and 2D Axial RoPE."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Ministral3.configs import PixtralVisionConfig
from Zoo.Common.RoPE import apply_rotary_pos_emb
from Zoo.Common.vision_utils import generate_block_attention_mask


class PixtralRMSNorm(nn.Module):
    """RMSNorm computed in FP32 for numerical stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(variance + self.variance_epsilon) * self.weight).to(orig_dtype)


class PixtralVisionRotaryEmbedding(nn.Module):
    """2D Continuous Axial Rotary Position Embedding matching official H-W-H-W layout."""
    inv_freq: torch.Tensor

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # inv_freq_2d: [inv_freq[0::2], inv_freq[1::2]]
        self.register_buffer("inv_freq", torch.cat([inv_freq[0::2], inv_freq[1::2]]), persistent=False)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates (cos, sin) tensors for 2D spatial coordinates pos_ids: (seq_len, 2)."""
        # pos_ids[:, 0:1] is height (h), pos_ids[:, 1:2] is width (w)
        half_dim = self.inv_freq.shape[0] // 2
        
        # Element-wise broadcasting for height and width axes
        freqs_h = pos_ids[:, 0:1].float() * self.inv_freq[:half_dim][None, :].float()  # (seq_len, 16)
        freqs_w = pos_ids[:, 1:2].float() * self.inv_freq[half_dim:][None, :].float()  # (seq_len, 16)
        
        # Interleave as [H, W] -> length 32
        freq_hw = torch.cat([freqs_h, freqs_w], dim=-1)
        emb = torch.cat([freq_hw, freq_hw], dim=-1)
        
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class PixtralAttention(nn.Module):
    """Pixtral Multi-Head Self-Attention powered by PyTorch SDPA."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        if pos_emb is not None:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)

        # Fused FlashAttention / Memory-Efficient SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, s, -1))


class PixtralMLP(nn.Module):
    """Standard SwiGLU FFN."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PixtralAttentionLayer(nn.Module):
    """Transformer Encoder block."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size)
        self.attention = PixtralAttention(config)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size)
        self.feed_forward = PixtralMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), attn_mask=mask, pos_emb=pos)
        return x + self.feed_forward(self.ffn_norm(x))


class PixtralTransformer(nn.Module):
    """Stack of PixtralAttentionLayers."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([PixtralAttentionLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, pos=pos)
        return x


class PixtralVisionModel(nn.Module):
    """Pixtral Vision Tower matching Hugging Face weights 1:1."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_conv = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size, bias=False,
        )
        self.ln_pre = PixtralRMSNorm(config.hidden_size)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralVisionRotaryEmbedding(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Processes images with arbitrary aspect ratios into a packed visual token sequence."""
        if image_sizes is None:
            b, _, h, w = pixel_values.shape
            image_sizes = torch.tensor([[h, w]] * b, device=pixel_values.device)

        patches = self.patch_conv(pixel_values.to(dtype=self.patch_conv.weight.dtype))
        patch_list, coords_list, patch_counts = [], [], []

        for p_tensor, size in zip(patches, image_sizes):
            hp = int(size[0].item()) // self.patch_size
            wp = int(size[1].item()) // self.patch_size
            patch_list.append(p_tensor[..., :hp, :wp].flatten(1).T)
            patch_counts.append(hp * wp)

            h_c, w_c = torch.meshgrid(
                torch.arange(hp, device=pixel_values.device),
                torch.arange(wp, device=pixel_values.device),
                indexing="ij",
            )
            coords_list.append(torch.stack([h_c.flatten(), w_c.flatten()], dim=-1))

        # (1, total_patches, hidden_size)
        embeds = self.ln_pre(torch.cat(patch_list, dim=0).unsqueeze(0))
        cos, sin = self.patch_positional_embedding(embeds, torch.cat(coords_list, dim=0))
        mask = generate_block_attention_mask(patch_counts, device=embeds.device, as_boolean=True)

        return self.transformer(embeds, mask=mask, pos=(cos, sin))