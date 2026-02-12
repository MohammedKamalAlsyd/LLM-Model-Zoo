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
from typing import Any, Tuple

import torch
from torch import nn

__all__ = [
    "SigLipVisionConfig",
    "SigLipVisionEmbeddings",
    "SigLipAttention",
    "SigLipMLP",
    "SigLipEncoderLayer",
    "SigLipEncoder",
    "SigLipVisionTransformer",
    "SigLipVisionModel",
]


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
    projection_dim: int = 2304
    num_channels: int = 3
    image_size: int = 224
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
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


class SigLipAttention(nn.Module):
    """Multi-head scaled dot-product attention module.

    Implements standard scaled dot-product attention with multiple heads,
    projecting input onto query, key, and value spaces before computing attention.

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        hidden_state (torch.Tensor): Input tensor of shape `(B, seq_len, embed_dim)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Attention output of shape `(B, seq_len, embed_dim)`
            and attention weights of shape `(B, num_heads, seq_len, seq_len)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.embed_dim // self.num_heads
        self.scale: float = self.head_dim**-0.5
        self.dropout: float = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_state: torch.Tensor  # (B, seq_len, embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, seq_len, _ = hidden_state.size()

        # Project to QKV
        q = self.q_proj(hidden_state)  # (B, seq_len, embed_dim)
        k = self.k_proj(hidden_state)  # (B, seq_len, embed_dim)
        v = self.v_proj(hidden_state)  # (B, seq_len, embed_dim)

        # Reshape for multi-head: (B, num_heads, seq_len, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (B, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(
            attn_weights, dim=-1
        )  # (B, num_heads, seq_len, seq_len)

        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, seq_len, head_dim)

        # Concatenate heads and project out
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)  # (B, seq_len, embed_dim)

        return output, attn_weights


class SigLipMLP(nn.Module):
    """Feed-forward network (MLP) module.

    A two-layer MLP with GELU activation, used as the feed-forward
    component in transformer encoder layers.

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        hidden_state (torch.Tensor): Input tensor of shape `(B, seq_len, embed_dim)`.

    Returns:
        torch.Tensor: Output tensor of shape `(B, seq_len, embed_dim)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim: int = config.hidden_size
        self.intermediate_dim: int = config.intermediate_size

        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_dim)
        self.fc2 = nn.Linear(self.intermediate_dim, self.embed_dim)
        self.activation = nn.GELU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute MLP forward pass."""
        x = self.fc1(hidden_state)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SigLipEncoderLayer(nn.Module):
    """Single transformer encoder layer with attention and feed-forward networks.

    Combines multi-head self-attention with an MLP in a residual configuration
    using layer normalization (pre-norm architecture).

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        hidden_state (torch.Tensor): Input tensor of shape `(B, seq_len, embed_dim)`.

    Returns:
        torch.Tensor: Output tensor of shape `(B, seq_len, embed_dim)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.attention = SigLipAttention(config)
        self.mlp = SigLipMLP(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute encoder layer forward pass with pre-normalization."""
        # Self-attention block with residual connection
        attn_output, _ = self.attention(self.layer_norm1(hidden_state))
        hidden_state = hidden_state + attn_output

        # MLP block with residual connection
        mlp_output = self.mlp(self.layer_norm2(hidden_state))
        hidden_state = hidden_state + mlp_output

        return hidden_state


class SigLipEncoder(nn.Module):
    """Stack of transformer encoder layers.

    Applies a sequence of transformer encoder layers to progressively
    refine the token representations through self-attention and feed-forward.

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        hidden_state (torch.Tensor): Input tensor of shape `(B, seq_len, embed_dim)`.

    Returns:
        torch.Tensor: Output tensor of shape `(B, seq_len, embed_dim)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Pass input through all encoder layers sequentially."""
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class SigLipVisionTransformer(nn.Module):
    """Vision transformer backbone for patch-based image encoding.

    Combines patch embeddings with positional information, applies a stack
    of transformer encoder layers, and includes post-normalization for stable
    feature outputs.

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        pixel_values (torch.Tensor): Input images of shape `(B, C, H, W)`.

    Returns:
        torch.Tensor: Encoded patch embeddings of shape `(B, num_patches, embed_dim)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim: int = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into patch embeddings through the vision transformer."""
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        return x


class SigLipVisionModel(nn.Module):
    """High-level vision model wrapper for SigLip vision transformer.

    Provides a unified interface to the vision transformer, encapsulating
    the complete image encoding pipeline.

    Args:
        config: SigLipVisionConfig instance with model hyperparameters.

    Forward Args:
        pixel_values (torch.Tensor): Input images of shape `(B, C, H, W)`.

    Returns:
        torch.Tensor: Encoded patch embeddings of shape `(B, num_patches, embed_dim)`.
    """

    def __init__(self, config: SigLipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode input images into patch embeddings."""
        return self.vision_model(pixel_values)
