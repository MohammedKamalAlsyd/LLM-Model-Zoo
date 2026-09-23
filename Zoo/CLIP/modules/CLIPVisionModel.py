"""CLIP Vision Transformer Tower with patch embedding and CLS summary projection."""

import torch
import torch.nn as nn
from Zoo.CLIP.configs import CLIPVisionConfig
from Zoo.CLIP.modules.CLIPEncoder import CLIPEncoder


class CLIPVisionEmbeddings(nn.Module):
    """Converts 2D pixel values into flattened patch tokens with CLS prefix."""

    def __init__(self, cfg: CLIPVisionConfig) -> None:
        super().__init__()
        self.embed_dim = cfg.hidden_size
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=cfg.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Args:

            pixel_values: FloatTensor of shape (batch_size, 3, height, width).

        Returns:
            Tensor of shape (batch_size, num_patches + 1, hidden_size).
        """
        b = pixel_values.shape[0]
        # (B, 3, H, W) -> (B, hidden_size, Grid_H, Grid_W) -> (B, hidden_size, Num_Patches) -> (B, Num_Patches, hidden_size)
        patch_embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)

        # Prepend [CLS] token embedding
        class_embeds = self.class_embedding.expand(b, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        return embeddings + self.position_embedding(self.position_ids)


class CLIPVisionTransformer(nn.Module):
    """Core Vision Transformer preserving exact `vision_model.*` checkpoint hierarchy."""

    def __init__(self, cfg: CLIPVisionConfig) -> None:
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(cfg)
        self.pre_layrnorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.encoder = CLIPEncoder(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads,
            num_layers=cfg.num_hidden_layers,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Args:

            pixel_values: FloatTensor of shape (batch_size, 3, height, width).

        Returns:
            Pooled [CLS] feature vector of shape (batch_size, hidden_size).
        """
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(hidden_states, causal_attention_mask=False)

        # Extract pooled [CLS] representation (index 0)
        pooled_output = hidden_states[:, 0, :]
        return self.post_layernorm(pooled_output)


class CLIPVisionModel(nn.Module):
    """Top-level Vision Model matching standard Hugging Face checkpoints."""

    def __init__(self, cfg: CLIPVisionConfig) -> None:
        super().__init__()
        self.vision_model = CLIPVisionTransformer(cfg)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)