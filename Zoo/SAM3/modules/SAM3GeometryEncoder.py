"""Geometry Prompt Encoder for bounding boxes with RoI-Align and coordinate embeddings."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.ops

from Zoo.Common.vision_utils import box_cxcywh_to_xyxy
from Zoo.SAM3.modules.SAM3Common import (
    Sam3Attention,
    Sam3MLP,
    concat_padded_sequences,
)


class Sam3GeometrySineEmbedding(nn.Module):
    """Coordinate-specific sine-cosine positional encoder for bounding box prompt geometry."""

    def __init__(self, num_position_features: int = 128, temperature: float = 10000.0) -> None:
        super().__init__()
        self.num_pos_feats = num_position_features
        self.temperature = temperature
        self.scale = 2.0 * torch.pi

    def encode_1d_positions(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = (x * self.scale)[:, None] / dim_t
        pos_y = (y * self.scale)[:, None] / dim_t
        px = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        py = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return px, py


class Sam3GeometryEncoderLayer(nn.Module):
    """Self-attention across prompts followed by cross-attention to visual feature maps."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 2048, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = Sam3Attention(hidden_size, num_heads)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = Sam3Attention(hidden_size, num_heads)

        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.mlp = Sam3MLP(hidden_size, intermediate_size, act="relu", dropout=dropout)

    def forward(
        self,
        prompt_feats: torch.Tensor,
        vision_feats: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Prompt Self-Attention
        residual = prompt_feats
        h = self.layer_norm1(prompt_feats)
        h, _ = self.self_attn(h, h, h, attention_mask=prompt_mask)
        prompt_feats = residual + self.dropout(h)

        # 2. Vision Cross-Attention
        residual = prompt_feats
        h = self.layer_norm2(prompt_feats)
        key = vision_feats + vision_pos_encoding
        h, _ = self.cross_attn(h, key, vision_feats)
        prompt_feats = residual + self.dropout(h)

        # 3. FFN Residual
        return prompt_feats + self.dropout(self.mlp(self.layer_norm3(prompt_feats)))


class Sam3GeometryEncoder(nn.Module):
    """Processes box prompts into high-dimensional tokens injected into DETR."""

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 2048,
        num_layers: int = 3,
        num_heads: int = 8,
        roi_size: int = 7,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.roi_size = roi_size

        self.position_encoding = Sam3GeometrySineEmbedding(num_position_features=hidden_size // 2)
        self.label_embed = nn.Embedding(2, hidden_size)
        self.cls_embed = nn.Embedding(1, hidden_size)

        # Projections
        self.boxes_direct_project = nn.Linear(4, hidden_size)
        self.boxes_pool_project = nn.Conv2d(hidden_size, hidden_size, roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden_size + 2, hidden_size)

        self.vision_layer_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        self.prompt_layer_norm = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList([
            Sam3GeometryEncoderLayer(hidden_size, intermediate_size, num_heads)
            for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    def _encode_boxes(
        self,
        boxes: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_boxes = boxes.shape[:2]
        height, width = vision_features.shape[-2:]

        # Direct coordinate embedding
        boxes_embed = self.boxes_direct_project(boxes)

        # RoI Align pooling over fine vision features
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).view(1, 1, 4)
        boxes_xyxy_scaled = boxes_xyxy * scale

        dtype = torch.float16 if vision_features.dtype == torch.bfloat16 else vision_features.dtype
        sampled_features = torchvision.ops.roi_align(
            vision_features.to(dtype),
            list(boxes_xyxy_scaled.to(dtype).unbind(0)),
            self.roi_size,
        ).to(vision_features.dtype)

        pooled_proj = self.boxes_pool_project(sampled_features).view(batch_size, num_boxes, self.hidden_size)
        boxes_embed = boxes_embed + pooled_proj

        # Center sine-encoding + box dimensions
        cx, cy, bw, bh = boxes.unbind(-1)
        pos_x, pos_y = self.position_encoding.encode_1d_positions(cx.flatten(), cy.flatten())
        pos_enc = torch.cat([pos_y, pos_x, bh.flatten()[:, None], bw.flatten()[:, None]], dim=1)
        boxes_embed = boxes_embed + self.boxes_pos_enc_project(pos_enc.view(batch_size, num_boxes, -1))

        # Positive / negative label embeddings
        boxes_embed = boxes_embed + self.label_embed(box_labels.long())
        return boxes_embed, box_mask

    def forward(
        self,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
        img_feats: Tuple[torch.Tensor, ...],
        img_pos_embeds: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = box_embeddings.shape[0]

        vision_feats = img_feats[-1]
        vision_pos = img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(vision_feats)

        vision_feats_flat = vision_feats.flatten(2).transpose(1, 2)
        vision_pos_flat = vision_pos.flatten(2).transpose(1, 2)

        norm_img_feats = self.vision_layer_norm(vision_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        prompt_embeds, prompt_mask = self._encode_boxes(box_embeddings, box_mask, box_labels, norm_img_feats)

        # Prepend global CLS token
        cls_embed = self.cls_embed.weight.view(1, 1, self.hidden_size).expand(batch_size, -1, -1)
        cls_mask = torch.ones(batch_size, 1, dtype=prompt_mask.dtype, device=prompt_mask.device)
        prompt_embeds, prompt_mask = concat_padded_sequences(prompt_embeds, prompt_mask, cls_embed, cls_mask)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        attn_mask = None
        if prompt_mask is not None:
            attn_mask = torch.where(prompt_mask[:, None, None, :], 0.0, float("-inf"))

        for layer in self.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_flat,
                prompt_mask=attn_mask,
            )

        return self.output_layer_norm(prompt_embeds), prompt_mask