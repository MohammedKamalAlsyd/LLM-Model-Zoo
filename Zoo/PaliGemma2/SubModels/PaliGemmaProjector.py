"""PaliGemma Multi-Modal Projector Module.

This module provides a linear projection layer to align image feature embeddings
from the vision encoder with the language model's embedding space. This is essential
for effective multi-modal integration in the PaliGemma2 model.

Classes:
    PaliGemmaMultiModalProjector: Linear projection module for feature alignment.
"""

import torch
from torch import nn

__all__ = ["PaliGemmaMultiModalProjector"]


class PaliGemmaMultiModalProjector(nn.Module):
    """Linear projection layer for aligning vision and language embeddings.

    Projects image patch embeddings from the vision encoder to the language model's
    embedding dimension, enabling seamless integration of visual and textual information.

    This projector ensures that image features are in the same semantic space as
    text token embeddings, facilitating cross-modal attention and fusion in the
    multi-modal transformer architecture.

    Attributes:
        hidden_size (int): Output projection dimension (language model embedding size).
        vision_projection_dim (int): Input dimension from vision encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        vision_projection_dim: int,
    ) -> None:
        """Initialize the multi-modal projector.

        Args:
            hidden_size (int): Output projection dimension for language model embeddings.
                Should match the language model's hidden/embedding size.
            vision_projection_dim (int): Input dimension from the vision encoder's
                projection output.

        Raises:
            ValueError: If hidden_size or vision_projection_dim are <= 0.
            TypeError: If arguments are not integers.
        """
        super().__init__()

        if not isinstance(hidden_size, int):
            raise TypeError(f"hidden_size must be int, got {type(hidden_size)}")
        if not isinstance(vision_projection_dim, int):
            raise TypeError(
                f"vision_projection_dim must be int, got {type(vision_projection_dim)}"
            )

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if vision_projection_dim <= 0:
            raise ValueError(
                f"vision_projection_dim must be positive, got {vision_projection_dim}"
            )

        self.hidden_size = hidden_size
        self.vision_projection_dim = vision_projection_dim

        # Linear projection layer: maps vision features to language model space
        self.linear = nn.Linear(
            in_features=vision_projection_dim,
            out_features=hidden_size,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project image features to language model embedding space.

        Applies a linear transformation to align vision patch embeddings with
        the language model's embedding dimension.

        Args:
            image_features (torch.Tensor): Image patch embeddings of shape
                `(batch_size, num_patches, vision_projection_dim)`.

        Returns:
            torch.Tensor: Projected embeddings of shape
                `(batch_size, num_patches, hidden_size)` aligned with the
                language model embedding space.

        Raises:
            RuntimeError: If input tensor shape is incompatible.

        Shape:
            - Input: `(B, N, vision_projection_dim)` where B is batch size,
                N is number of patches.
            - Output: `(B, N, hidden_size)`
        """
        if image_features.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch, patches, features), "
                f"got {image_features.dim()}D tensor of shape {image_features.shape}"
            )

        if image_features.shape[-1] != self.vision_projection_dim:
            raise ValueError(
                f"Input feature dimension {image_features.shape[-1]} does not match "
                f"expected vision_projection_dim {self.vision_projection_dim}"
            )

        # Project to language model space
        # Input:  (batch_size, num_patches, vision_projection_dim)
        # Output: (batch_size, num_patches, hidden_size)
        projected_features = self.linear(image_features)

        return projected_features
