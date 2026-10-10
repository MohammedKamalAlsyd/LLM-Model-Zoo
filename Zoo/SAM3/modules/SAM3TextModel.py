"""Compact SAM3 Text Encoder reusing universal CLIP infrastructure."""

from typing import Optional
import torch
import torch.nn as nn

from Zoo.CLIP.modules.CLIPEncoder import CLIPEncoder
from Zoo.CLIP.modules.CLIPTextModel import CLIPTextEmbeddings
from Zoo.SAM3.configs import SAM3TextConfig


class CLIPTextTransformer(nn.Module):
    """Text Transformer hierarchy `text_encoder.text_model.*` using standard GELU."""

    def __init__(self, cfg: SAM3TextConfig) -> None:
        super().__init__()
        # 1. Type-safe reuse of CLIPTextEmbeddings
        self.embeddings = CLIPTextEmbeddings(cfg)
        
        # 2. Universal encoder with standard GELU
        self.encoder = CLIPEncoder(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads,
            num_layers=cfg.num_hidden_layers,
            layer_norm_eps=cfg.layer_norm_eps,
            act="gelu",
        )
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        seq_len = input_ids.shape[-1]
        device, dtype = input_ids.device, hidden_states.dtype

        # Combined Causal + Padding Attention Mask
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]

        if attention_mask is not None:
            pad_mask = torch.where(attention_mask[:, None, None, :].bool(), 0.0, float("-inf")).to(dtype)
            combined_mask = causal_mask + pad_mask
        else:
            combined_mask = causal_mask

        hidden_states = self.encoder(hidden_states, attention_mask=combined_mask)
        return self.final_layer_norm(hidden_states)


class Sam3TextEncoder(nn.Module):
    """Top-level Text Encoder matching checkpoint keys `text_encoder.*`."""

    def __init__(self, cfg: Optional[SAM3TextConfig] = None) -> None:
        super().__init__()
        self.config = cfg or SAM3TextConfig()
        self.text_model = CLIPTextTransformer(self.config)
        self.text_projection = nn.Linear(self.config.hidden_size, self.config.projection_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.text_model(input_ids, attention_mask=attention_mask)