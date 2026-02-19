"""
Mistral3 multimodal projector helpers.

This module contains small utilities to merge vision patch tokens and
project them into the text embedding space used by Mistral3. Important:
do not rename or remove any layer attributes (for example
`merging_layer`, `patch_merger`, `linear_1`, `linear_2`, `norm`) because
the user intends to load weights copied from Hugging Face that rely on
these exact names.

Expected shapes (high level):
- `image_features`: flat patch tokens concatenated across batch; last dim is embed_dim
- `image_sizes`: sequence of per-image grid sizes or pixel sizes used to reconstruct grids

The code intentionally keeps attribute names used when saving/loading
weights to remain fully compatible with external checkpoints.
"""

from typing import Iterable

import torch
from torch import nn
from Mistral3 import Mistral3RMSNorm, Mistral3Config


class Mistral3PatchMerger(nn.Module):
    """Learned merging of small spatial patches into larger patches.

    The module collects groups of `spatial_merge_size**2` neighboring patches
    and applies a single linear layer (`merging_layer`) to reduce the
    concatenated embeddings back to `hidden_size`.

    Notes for users copying weights from Hugging Face:
    - Keep the attribute name `merging_layer` as-is (it maps to checkpoint keys).
    - Do not change `spatial_merge_size` or `patch_size` semantics here.
    """

    def __init__(self, config: Mistral3Config) -> None:
        super().__init__()

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size

        # A learned linear projection that reduces concatenated patch vectors
        # (embed_dim * merge_size**2) -> hidden_size. Keep name for checkpointing.
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Merge image patch tokens into larger patches.

        Args:
            image_features: concatenated patch tokens for all images in batch
                (shape: [total_patches, embed_dim]). The tensor is expected to
                be a concatenation of per-image token sequences.
            image_sizes: iterable of per-image sizes used to reshape tokens.
                Each entry should be convertible to (H_patches, W_patches)
                so that H_patches * W_patches equals the number of tokens for
                that image.

        Returns:
            Tensor with merged patches projected to `hidden_size`.
        """

        # Compute how many patch tokens each image contributes (in patch-grid units)
        patch_grid_sizes = [
            (
                int(image_size[0]) // self.patch_size,
                int(image_size[1]) // self.patch_size,
            )
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in patch_grid_sizes]
        embed_dim = image_features.shape[-1]

        # We'll accumulate per-image merged patch blocks and concatenate later.
        permuted_tensor = []

        # Split the flat image_features into per-image token tensors
        for image_index, image_tokens in enumerate(
            image_features.split(tokens_per_image)
        ):
            # Reconstruct the 2D grid of patches for this image
            # `image_sizes` may contain pixel dims, but we already converted
            # them into patch-grid dims above; reuse those dims here to reshape.
            h_patches, w_patches = patch_grid_sizes[image_index]

            # image_tokens: (H_patches * W_patches, embed_dim)
            # reshape -> (H_patches, W_patches, embed_dim)
            image_grid = image_tokens.view(h_patches, w_patches, embed_dim)

            # Move channel/embed dim to front and add batch dim for unfold:
            # (1, embed_dim, H_patches, W_patches)
            image_grid = image_grid.permute(2, 0, 1).unsqueeze(0)

            # Use unfold to extract non-overlapping blocks of size spatial_merge_size
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )

            # grid shape -> (1, embed_dim * merge_size**2, N_windows)
            # Reshape to (N_windows, embed_dim * merge_size**2) so we can apply
            # the linear merging layer across the last dim.
            grid = grid.view(embed_dim * self.spatial_merge_size**2, -1).t()

            permuted_tensor.append(grid)

        # Concatenate merged blocks for all images and apply the learned linear layer
        merged_patches = torch.cat(permuted_tensor, dim=0)
        merged_patches = self.merging_layer(merged_patches)
        return merged_patches


class Mistral3MultiModalProjector(nn.Module):
    """Project vision tokens into Mistral3 text embedding space.

    This thin module performs the following steps in order:
    1. RMS normalization (`norm`) on incoming vision features.
    2. Merge neighboring patches with `patch_merger`.
    3. Two linear layers with GELU non-linearity (`linear_1`, `act`, `linear_2`)
       to map vision `hidden_size` -> text `hidden_size`.

    All layer attribute names are preserved to allow direct weight loading
    from checkpoints.
    """

    def __init__(self, config: Mistral3Config) -> None:
        super().__init__()

        # Normalize vision features before merging/projection
        self.norm = Mistral3RMSNorm(
            config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps
        )

        # Patch merger keeps its attribute name to match checkpoints
        self.patch_merger = Mistral3PatchMerger(config)

        # Two projection layers with a GELU activation in between.
        # Keep layer names `linear_1` and `linear_2` for checkpoint compatibility.
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=False
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass: normalize, merge, and project image patches.

        Args:
            image_features: concatenated patch tokens for the batch
                (shape: [total_patches, vision_hidden_size]).
            image_sizes: iterable with per-image sizes used by the patch merger.

        Returns:
            Tensor: projected patches in text embedding space
                (shape: [total_merged_patches, text_hidden_size]).
        """

        # RMS normalize incoming features
        normed_patches = self.norm(image_features)

        # Merge spatial patches into larger patches
        merged_patches = self.patch_merger(normed_patches, image_sizes)

        # Project merged vision patches into text hidden space
        projected_patches = self.linear_1(merged_patches)
        projected_patches = self.act(projected_patches)
        projected_patches = self.linear_2(projected_patches)

        return projected_patches
