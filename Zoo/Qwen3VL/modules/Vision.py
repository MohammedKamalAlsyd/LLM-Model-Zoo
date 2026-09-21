"""Qwen3-VL Vision Encoder with DeepStack Feature Fusion and 2D Axial RoPE."""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Common.RoPE import apply_rotary_pos_emb
from Zoo.Qwen3VL.configs import Qwen3VLVisionConfig
from Zoo.Common.vision_utils import get_vision_position_ids, get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights

# ============================================================================
# Rotary Position Embedding & Patch Processing
# ============================================================================

class VisionRotaryEmbedding(nn.Module):
    """2D Axial Rotary Position Embedding for non-square patch layouts."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates (cos, sin) tensors for 2D spatial positions.

        Args:
            position_ids: Tensor of shape (total_tokens, 2) containing [H, W] coordinates.

        Returns:
            cos, sin: Frequency tensors shaped for head broadcasting (total_tokens, 1, head_dim).
        """
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq[None, None, :]
        cos = freqs.cos()
        sin = freqs.sin()

        cos_hw = torch.cat([cos[:, 0], cos[:, 1]], dim=-1)
        sin_hw = torch.cat([sin[:, 0], sin[:, 1]], dim=-1)

        cos_full = torch.cat([cos_hw, cos_hw], dim=-1).unsqueeze(1)
        sin_full = torch.cat([sin_hw, sin_hw], dim=-1).unsqueeze(1)
        return cos_full, sin_full


class VisionPatchEmbed(nn.Module):
    """3D Conv patchification for spatio-temporal video and image processing."""

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel, stride=kernel, bias=True)
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects image/video tensor into a flattened patch token stream.

        Args:
            x: Input pixel values tensor.

        Returns:
            Tensor of shape (total_patches, hidden_size).
        """
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


# ============================================================================
# Vision Transformer Blocks
# ============================================================================

class VisionMLP(nn.Module):
    """Two-layer Feed-Forward Network with GeLU approximation."""

    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionAttention(nn.Module):
    """Multi-Head Attention supporting variable-length packed sequence chunks."""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        q, k, v = (
            self.qkv(x)
            .reshape(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Process each variable length visual segment independently with SDPA
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(q, lengths, dim=0)
        k_splits = torch.split(k, lengths, dim=0)
        v_splits = torch.split(v, lengths, dim=0)

        out_splits = []
        for q_c, k_c, v_c in zip(q_splits, k_splits, v_splits):
            q_c = q_c.transpose(0, 1).unsqueeze(0)  # (1, num_heads, chunk_len, head_dim)
            k_c = k_c.transpose(0, 1).unsqueeze(0)
            v_c = v_c.transpose(0, 1).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(q_c, k_c, v_c, is_causal=False)
            out_splits.append(attn_out.squeeze(0).transpose(0, 1))

        out = torch.cat(out_splits, dim=0).reshape(seq_len, -1)
        return self.proj(out)


class VisionBlock(nn.Module):
    """Transformer Encoder layer with Pre-LayerNorm residual connections."""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, intermediate_size: int = 4096) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, cos=cos, sin=sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Patch Mergers & Top-Level Vision Tower
# ============================================================================

class VisionPatchMerger(nn.Module):
    """Downsamples spatial 2x2 patch neighborhoods and projects to LLM hidden size."""

    def __init__(
        self,
        hidden_size: int = 1024,
        out_hidden_size: int = 2560,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.merged_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.merged_size if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.linear_fc2 = nn.Linear(self.merged_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL Vision Tower with DeepStack intermediate representation taps."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)

        self.patch_embed = VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=3,
            hidden_size=config.hidden_size,
        )
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim=config.hidden_size // config.num_heads)

        self.blocks = nn.ModuleList([
            VisionBlock(config.hidden_size, config.num_heads, config.intermediate_size)
            for _ in range(config.depth)
        ])

        # Final projection merger
        self.merger = VisionPatchMerger(
            hidden_size=config.hidden_size,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            use_postshuffle_norm=False,
        )

        # DeepStack intermediate layer projection taps
        self.deepstack_merger_list = nn.ModuleList([
            VisionPatchMerger(
                hidden_size=config.hidden_size,
                out_hidden_size=config.out_hidden_size,
                spatial_merge_size=config.spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ])

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Processes images/videos into final visual tokens and DeepStack representations.

        Args:
            pixel_values: Flattened patch pixel values.
            grid_thw: Tensor of shape (num_items, 3) holding [T, H, W] grid sizes.

        Returns:
            pooler_output: Downsampled visual tokens (num_tokens, out_hidden_size).
            deepstack_features: List of intermediate features for language layer injection.
        """
        interp_idx, interp_wt = get_vision_interpolation_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=False)

        x = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(interp_idx) * interp_wt.unsqueeze(-1)).sum(dim=1)
        x = x + pos_embeds.to(x.dtype)

        cos, sin = self.rotary_pos_emb(position_ids)

        deepstack_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            if i in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](x))

        merged_tokens = self.merger(x)
        return merged_tokens, deepstack_features