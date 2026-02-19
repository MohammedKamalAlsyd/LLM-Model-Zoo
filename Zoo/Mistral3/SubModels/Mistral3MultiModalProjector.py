from typing import Any

import torch
from torch import nn
from Mistral3 import Mistral3RMSNorm, Mistral3Config


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config) -> None:
        super().__init__()

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )  # Input Shape: (B, H, W, hidden_size * spatial_merge_size**2), Output Shape: (B, H//spatial_merge_size, W//spatial_merge_size, hidden_size)

    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        """Merges spatial_merge_size ** 2 patches into one patch using a learned linear layer."""
        patch_grid_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in patch_grid_sizes]
        embed_dim = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(
            image_features.split(tokens_per_image)
        ):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            h, w = int(h), int(w)

            image_grid = (
                image_tokens.view(h, w, embed_dim).permute(2, 0, 1).unsqueeze(0)
            )  # Input Shape: (B, H*W, embed_dim), Output Shape: (B, embed_dim, H, W)

            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )  # Input Shape: (B, embed_dim, H, W), Output Shape: (B, embed_dim * spatial_merge_size**2, H//spatial_merge_size * W//spatial_merge_size)

            grid = grid.view(embed_dim * self.spatial_merge_size**2, -1).t()
            # Input Shape: (B, embed_dim * spatial_merge_size**2, H//spatial_merge_size * W//spatial_merge_size), Output Shape: (B * H//spatial_merge_size * W//spatial_merge_size, embed_dim * spatial_merge_size**2)
            # in other words each grid item become now d*merge_size**2 and we have B*H//merge_size*W//merge_size of them
            permuted_tensor.append(grid)

        merged_patches = torch.cat(
            permuted_tensor, dim=0
        )  # Shape: (total_merged_patches, embed_dim * spatial_merge_size**2)
        merged_patches = self.merging_layer(
            merged_patches
        )  # Shape: (total_merged_patches, hidden_size)
        return merged_patches


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config) -> None:
        super().__init__()
        self.norm = Mistral3RMSNorm(
            config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.patch_merger = Mistral3PatchMerger(config)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=False
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        normed_patches = self.norm(
            image_features
        )  # Shape: (total_merged_patches, hidden_size)
        merged_patches = self.patch_merger(
            normed_patches, image_sizes
        )  # Shape: (total_merged_patches, hidden_size)
        projected_patches = self.linear_1(
            merged_patches
        )  # Shape: (total_merged_patches, text_hidden_size)
        projected_patches = self.act(
            projected_patches
        )  # Shape: (total_merged_patches, text_hidden_size)
        projected_patches = self.linear_2(
            projected_patches
        )  # Shape: (total_merged_patches, text_hidden_size)
        return projected_patches
