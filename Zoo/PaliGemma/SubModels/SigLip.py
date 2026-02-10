"""SigLip submodule: vision patch embeddings.

This module provides a lightweight, well-documented implementation of
patch-based image embeddings suitable for transformer-style vision backbones.

Classes
- SigLipVisionConfig: dataclass holding model hyperparameters and validation.
- SigLipVisionEmbeddings: nn.Module that converts images to patch embeddings
  and adds learned positional embeddings.

Notes
- Input images are expected in PyTorch channel-first format: `(B, C, H, W)`.
- `image_size` must be evenly divisible by `patch_size`.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = ["SigLipVisionConfig", "SigLipVisionEmbeddings"]


@dataclass
class SigLipVisionConfig:
    """Configuration for SigLip vision components.

    Attributes
    - hidden_size: Dimensionality of token embeddings produced per patch.
    - intermediate_size: Hidden size for any downstream MLP (kept for compatibility).
    - num_attention_heads: Number of attention heads (compatibility only).
    - num_hidden_layers: Number of transformer layers (compatibility only).
    - patch_size: Spatial size (in pixels) of each square patch.
    - projection_dim: Optional projection dimension for downstream heads.
    - num_channels: Number of image channels (3 for RGB).
    - image_size: Expected input image height/width (must be square).
    """

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    projection_dim: int = 2048
    num_channels: int = 3
    image_size: int = 224
    extra: Any = None

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")


class SigLipVisionEmbeddings(nn.Module):
    """Patch embedding module with learned positional embeddings.

    The module performs a conv-based patch projection followed by addition of
    an embedding lookup for the patch positions.

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        pixel_values (torch.FloatTensor): Tensor of shape `(B, C, H, W)`.

    Returns:
        torch.FloatTensor: Patch embeddings of shape `(B, num_patches, embed_dim)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_size = int(config.patch_size)
        self.embed_dim = int(config.hidden_size)
        self.num_channels = int(config.num_channels)
        self.image_size = int(config.image_size)

        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side * self.num_patches_per_side

        # Conv2d projects each patch to `embed_dim` channels.
        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

        # Learned positional embeddings for each patch index.
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

        # Precompute patch indices as a persistent buffer (long dtype).
        self.register_buffer(
            "patch_indices",
            torch.arange(self.num_patches, dtype=torch.long)
            .reshape(1, self.num_patches)
            .expand(1, -1),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute patch embeddings for input images.

        Steps:
        1. Apply conv patch projection -> (B, embed_dim, P, P)
        2. Flatten spatial dims and transpose -> (B, num_patches, embed_dim)
        3. Add positional embeddings
        """

        if pixel_values.dim() != 4:
            raise ValueError("pixel_values must have shape (B, C, H, W)")

        # Project patches: (B, embed_dim, P, P)
        x = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # Add positional embeddings (broadcast over batch dim)
        pos = self.position_embedding(self.patch_indices)
        x = x + pos

        return x
