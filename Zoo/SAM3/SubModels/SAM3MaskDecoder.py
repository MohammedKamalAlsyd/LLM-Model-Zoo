import torch
import torch.nn as nn
import torch.nn.functional as F
from Zoo.SAM3.SubModels.SAM3Common import Sam3Attention

class Sam3PixelDecoder(nn.Module):
    """
    FPN Pixel Decoder matching checkpoint keys:
      - pixel_decoder.conv_layers.0..2
      - pixel_decoder.norms.0..2
    """
    def __init__(self, hidden_size: int = 256, num_stages: int = 3):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
            for _ in range(num_stages)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, hidden_size)
            for _ in range(num_stages)
        ])
        self.out_channels = hidden_size

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        prev = features[-1]
        for i, feat in enumerate(reversed(features[:-1])):
            prev = F.interpolate(prev, size=feat.shape[-2:], mode="nearest") + feat
            prev = F.relu(self.norms[i](self.conv_layers[i](prev)))
        return prev


class Sam3MaskEmbedder(nn.Module):
    """
    Query Mask Embedder matching checkpoint keys:
      - mask_embedder.layers.0..2
    """
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class Sam3MaskDecoder(nn.Module):
    """
    Mask Decoder matching checkpoint keys:
      - pixel_decoder.*
      - mask_embedder.*
      - instance_projection
      - semantic_projection
      - prompt_cross_attn.*
      - prompt_cross_attn_norm
    """
    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.pixel_decoder = Sam3PixelDecoder(hidden_size=hidden_size)
        self.mask_embedder = Sam3MaskEmbedder(hidden_size=hidden_size)

        self.instance_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.semantic_projection = nn.Conv2d(hidden_size, 1, kernel_size=1)

        self.prompt_cross_attn = Sam3Attention(hidden_size, num_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size)
        self.prompt_cross_attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        prompt_features: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_features is not None:
            residual = encoder_hidden_states
            h = self.prompt_cross_attn_norm(encoder_hidden_states)
            attn_mask = None
            if prompt_mask is not None:
                attn_mask = torch.where(prompt_mask[:, None, None, :], 0.0, float("-inf"))
            h, _ = self.prompt_cross_attn(h, prompt_features, prompt_features, attention_mask=attn_mask)
            encoder_hidden_states = residual + self.prompt_cross_attn_dropout(h)

        # Reshape encoder features to replace the finest feature in FPN
        feats = [f.clone() for f in backbone_features]
        b, h, w = feats[-1].shape[0], feats[-1].shape[-2], feats[-1].shape[-1]
        encoder_spatial = encoder_hidden_states[:, :h * w, :].transpose(1, 2).reshape(b, -1, h, w)
        feats[-1] = encoder_spatial

        # Decode multi-scale FPN features into fine resolution pixel embeddings
        pixel_embed = self.pixel_decoder(feats)

        # Instance mask prediction (Q queries dot product with pixel embeddings)
        instance_embeds = self.instance_projection(pixel_embed)
        query_embeds = self.mask_embedder(decoder_queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", query_embeds, instance_embeds)

        # Semantic mask prediction
        semantic_seg = self.semantic_projection(pixel_embed)

        return pred_masks, semantic_seg