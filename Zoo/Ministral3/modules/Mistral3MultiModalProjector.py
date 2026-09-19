"""Multimodal Projector for Ministral-3.

Normalizes, merges adjacent 2x2 vision patches, and projects visual representations
into the text hidden embedding space.
"""

from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Ministral3.configs import Ministral3MultimodalConfig


class Mistral3RMSNorm(nn.Module):
    """RMSNorm computed in FP32 precision for numerical stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight).to(orig_dtype)


class Mistral3PatchMerger(nn.Module):
    """Merges groups of (spatial_merge_size x spatial_merge_size) vision patches.

    Flattens spatial neighborhoods while keeping channel vectors contiguous,
    then projects them via a linear merging layer.
    """

    def __init__(self, config: Ministral3MultimodalConfig) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        vision_hidden = config.vision_config.hidden_size

        # Linear reduction from (vision_hidden * merge_size^2) -> vision_hidden
        self.merging_layer = nn.Linear(
            vision_hidden * (self.spatial_merge_size ** 2),
            vision_hidden,
            bias=False,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_sizes: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Merges patch tokens per image.

        Args:
            image_features: Concatenated patch tokens of shape (total_patches, vision_hidden).
            image_sizes: Iterable of (height, width) pixel tensors per image.

        Returns:
            Tensor of shape (total_merged_patches, vision_hidden).
        """
        patch_grid_sizes = [
            (int(size[0].item()) // self.patch_size, int(size[1].item()) // self.patch_size)
            for size in image_sizes
        ]
        tokens_per_image = [h * w for h, w in patch_grid_sizes]
        embed_dim = image_features.shape[-1]
        m = self.spatial_merge_size

        merged_blocks: List[torch.Tensor] = []

        # Process each image independently based on its original 2D grid
        for img_tokens, (hp, wp) in zip(image_features.split(tokens_per_image), patch_grid_sizes):
            # Reshape to 2D patch grid: (hp, wp, embed_dim)
            grid = img_tokens.view(hp, wp, embed_dim)

            # Group into non-overlapping spatial blocks of (m x m)
            # (hp // m, m, wp // m, m, embed_dim) -> (hp // m, wp // m, m, m, embed_dim)
            h_merged = hp // m
            w_merged = wp // m
            grid = grid.view(h_merged, m, w_merged, m, embed_dim).permute(0, 2, 1, 3, 4).contiguous()

            # Flatten each (m x m) patch neighborhood while keeping embed_dim contiguous
            grid = grid.view(h_merged * w_merged, m * m * embed_dim)
            merged_blocks.append(grid)

        # Concatenate across all images in batch and project
        merged_tensor = torch.cat(merged_blocks, dim=0)
        return self.merging_layer(merged_tensor)


class Mistral3MultiModalProjector(nn.Module):
    """Full Multimodal Projector pipeline: RMSNorm -> PatchMerger -> 2-layer MLP."""

    def __init__(self, config: Ministral3MultimodalConfig) -> None:
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        text_hidden = config.text_config.hidden_size

        self.norm = Mistral3RMSNorm(vision_hidden, eps=config.text_config.rms_norm_eps)
        self.patch_merger = Mistral3PatchMerger(config)
        self.linear_1 = nn.Linear(vision_hidden, text_hidden, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=False)

    def forward(
        self,
        image_features: torch.Tensor,
        image_sizes: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Projects vision encoder features into the language model embedding space.

        Args:
            image_features: Tensor of shape (total_patches, vision_hidden).
            image_sizes: Iterable of (height, width) tensors.

        Returns:
            Tensor of shape (total_merged_tokens, text_hidden).
        """
        x = self.norm(image_features)
        x = self.patch_merger(x, image_sizes)
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)