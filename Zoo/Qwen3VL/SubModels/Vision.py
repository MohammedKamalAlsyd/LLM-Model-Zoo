import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence


from ..utils.rotate_functions import VisionRotaryEmbedding, apply_rotary_emb
from ..utils.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, temporal_patch_size: int = 2, in_channels: int = 3, hidden_size: int = 1024):
        super().__init__()
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel, stride=kernel, bias=True)
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (total_patches, in_channels, temporal_patch_size, patch_size, patch_size) or flattened
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionAttention(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
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

        q, k = apply_rotary_emb(q, k, cos, sin)

        # Process each variable length chunk with SDPA
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(q, lengths, dim=0)
        k_splits = torch.split(k, lengths, dim=0)
        v_splits = torch.split(v, lengths, dim=0)

        out_splits = []
        for q_c, k_c, v_c in zip(q_splits, k_splits, v_splits):
            # q_c: (chunk_len, heads, head_dim) -> (1, heads, chunk_len, head_dim)
            q_c = q_c.transpose(0, 1).unsqueeze(0)
            k_c = k_c.transpose(0, 1).unsqueeze(0)
            v_c = v_c.transpose(0, 1).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(q_c, k_c, v_c, is_causal=False)
            out_splits.append(attn_out.squeeze(0).transpose(0, 1))

        out = torch.cat(out_splits, dim=0).reshape(seq_len, -1)
        return self.proj(out)


class VisionBlock(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, intermediate_size: int = 4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, cos=cos, sin=sin)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nn.Module):
    def __init__(self, hidden_size: int = 1024, out_hidden_size: int = 2560, spatial_merge_size: int = 2, use_postshuffle_norm: bool = False):
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
    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        out_hidden_size: int = 2560,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: Sequence[int] = (5, 11, 17),
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        self.deepstack_visual_indexes = list(deepstack_visual_indexes)

        self.patch_embed = VisionPatchEmbed(patch_size, temporal_patch_size, in_channels=3, hidden_size=hidden_size)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim=hidden_size // num_heads)

        self.blocks = nn.ModuleList([
            VisionBlock(hidden_size, num_heads, intermediate_size) for _ in range(depth)
        ])

        self.merger = VisionPatchMerger(hidden_size, out_hidden_size, spatial_merge_size, use_postshuffle_norm=False)
        self.deepstack_merger_list = nn.ModuleList([
            VisionPatchMerger(hidden_size, out_hidden_size, spatial_merge_size, use_postshuffle_norm=True)
            for _ in range(len(deepstack_visual_indexes))
        ])

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            pixel_values: flattened patch pixels
            grid_thw: (num_items, 3) where each row is [T, H, W]
        Returns:
            pooler_output: (num_merged_tokens, out_hidden_size)
            deepstack_features: list of (num_merged_tokens, out_hidden_size) tensors
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