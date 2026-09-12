import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Zoo.SAM3.SubModels.SAM3Common import (
    Sam3Attention,
    Sam3MLP,
    Sam3DecoderMLP,
    Sam3SinePositionEmbedding,
    box_cxcywh_to_xyxy,
    inverse_sigmoid,
)

# ============================================================================
# DETR Encoder
# ============================================================================

class Sam3DetrEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 2048, num_heads: int = 8, dropout: float = 0.1):
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
        vision_feats: torch.Tensor,
        prompt_feats: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Vision Self-Attention with 2D coordinates
        residual = vision_feats
        h = self.layer_norm1(vision_feats)
        q_k = h + vision_pos_encoding
        h, _ = self.self_attn(query=q_k, key=q_k, value=h)
        vision_feats = residual + self.dropout(h)

        # Cross Attention: Vision tokens attend to Text/Prompt tokens
        residual = vision_feats
        h = self.layer_norm2(vision_feats)
        h, _ = self.cross_attn(query=h, key=prompt_feats, value=prompt_feats, attention_mask=prompt_mask)
        vision_feats = residual + self.dropout(h)

        # MLP
        residual = vision_feats
        h = self.layer_norm3(vision_feats)
        return residual + self.dropout(self.mlp(h))


class Sam3DetrEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 2048, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            Sam3DetrEncoderLayer(hidden_size, intermediate_size, num_heads)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        vision_features: list[torch.Tensor],
        text_features: torch.Tensor,
        vision_pos_embeds: list[torch.Tensor],
        text_mask: torch.Tensor | None = None,
    ):
        feat_list, pos_list, spatial_shapes = [], [], []
        for feat, pos in zip(vision_features, vision_pos_embeds):
            h, w = feat.shape[-2:]
            spatial_shapes.append((h, w))
            feat_list.append(feat.flatten(2).transpose(1, 2))
            pos_list.append(pos.flatten(2).transpose(1, 2))

        features_flat = torch.cat(feat_list, dim=1)
        pos_flat = torch.cat(pos_list, dim=1)
        spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.long, device=features_flat.device)

        prompt_attn_mask = None
        if text_mask is not None:
            prompt_attn_mask = torch.where(text_mask[:, None, None, :], 0.0, float("-inf"))

        hidden_states = features_flat
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                prompt_feats=text_features,
                vision_pos_encoding=pos_flat,
                prompt_mask=prompt_attn_mask,
            )

        return hidden_states, pos_flat, text_features, spatial_shapes_tensor

# ============================================================================
# DETR Decoder
# ============================================================================

class Sam3DetrDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Sam3Attention(hidden_size, num_heads)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.text_cross_attn = Sam3Attention(hidden_size, num_heads)
        self.text_cross_attn_dropout = nn.Dropout(dropout)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.vision_cross_attn = Sam3Attention(hidden_size, num_heads)
        self.vision_cross_attn_dropout = nn.Dropout(dropout)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.mlp = Sam3MLP(hidden_size, intermediate_size, act="relu", dropout=dropout)
        self.mlp_layer_norm = nn.LayerNorm(hidden_size)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        vision_cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Prepend zero pos embedding for presence token at index 0
        query_pos = F.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)

        # 1. Query Self-Attention
        residual = hidden_states
        q_pos = hidden_states + query_pos
        h, _ = self.self_attn(q_pos, q_pos, hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + self.self_attn_dropout(h))

        # 2. Text Cross-Attention
        residual = hidden_states
        q_pos = hidden_states + query_pos
        h, _ = self.text_cross_attn(q_pos, text_features, text_features, attention_mask=text_mask)
        hidden_states = self.text_cross_attn_layer_norm(residual + self.text_cross_attn_dropout(h))

        # 3. Vision Cross-Attention with Relative Position Bias
        residual = hidden_states
        q_pos = hidden_states + query_pos
        k_pos = vision_features + vision_pos_encoding
        h, _ = self.vision_cross_attn(q_pos, k_pos, vision_features, attention_mask=vision_cross_attn_mask)
        hidden_states = self.vision_cross_attn_layer_norm(residual + self.vision_cross_attn_dropout(h))

        # 4. MLP
        residual = hidden_states
        h = self.mlp(hidden_states)
        return self.mlp_layer_norm(residual + self.mlp_dropout(h))


class Sam3DetrDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 2048,
        num_layers: int = 6,
        num_queries: int = 200,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_heads

        self.query_embed = nn.Embedding(num_queries, hidden_size)
        self.reference_points = nn.Embedding(num_queries, 4)

        self.presence_token = nn.Embedding(1, hidden_size)
        self.presence_head = Sam3DecoderMLP(hidden_size, hidden_size, 1, num_layers=3)
        self.presence_layer_norm = nn.LayerNorm(hidden_size)

        self.ref_point_head = Sam3DecoderMLP(2 * hidden_size, hidden_size, hidden_size, num_layers=2)
        self.box_rpb_embed_x = Sam3DecoderMLP(2, hidden_size, num_heads, num_layers=2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, hidden_size, num_heads, num_layers=2)

        self.box_head = Sam3DecoderMLP(hidden_size, hidden_size, 4, num_layers=3)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList([
            Sam3DetrDecoderLayer(hidden_size, intermediate_size, num_heads)
            for _ in range(num_layers)
        ])
        self.position_encoding = Sam3SinePositionEmbedding(num_position_features=hidden_size // 2, normalize=False)

    def _get_rpb_matrix(self, reference_boxes: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
        h, w = spatial_shape
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        b, q, _ = boxes_xyxy.shape

        coords_h = torch.arange(0, h, device=reference_boxes.device, dtype=reference_boxes.dtype) / h
        coords_w = torch.arange(0, w, device=reference_boxes.device, dtype=reference_boxes.dtype) / w

        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(b, q, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(b, q, -1, 2)

        deltas_x_log = torch.sign(deltas_x * 8) * torch.log2(torch.abs(deltas_x * 8) + 1.0) / math.log2(8)
        deltas_y_log = torch.sign(deltas_y * 8) * torch.log2(torch.abs(deltas_y * 8) + 1.0) / math.log2(8)

        dx = self.box_rpb_embed_x(deltas_x_log)  # [B, Q, W, Heads]
        dy = self.box_rpb_embed_y(deltas_y_log)  # [B, Q, H, Heads]

        rpb = dy.unsqueeze(3) + dx.unsqueeze(2)  # [B, Q, H, W, Heads]
        rpb = rpb.flatten(2, 3).permute(0, 3, 1, 2).contiguous()  # [B, Heads, Q, H*W]
        return rpb

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ):
        batch_size = vision_features.shape[0]

        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1).sigmoid()
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)

        hidden_states = torch.cat([presence_token, query_embeds], dim=1)

        text_attn_mask = None
        if text_mask is not None:
            text_attn_mask = torch.where(text_mask[:, None, None, :], 0.0, float("-inf"))

        intermediate_outputs = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence = []

        for layer in self.layers:
            query_sine = self.position_encoding.encode_boxes(reference_boxes)
            query_pos = self.ref_point_head(query_sine)

            # RPB bias mask
            vision_attn_mask = None
            if spatial_shapes is not None and spatial_shapes.shape[0] == 1:
                h_dim = int(spatial_shapes[0, 0].item())
                w_dim = int(spatial_shapes[0, 1].item())
                rpb = self._get_rpb_matrix(reference_boxes, (h_dim, w_dim))
                # Pad zero row for presence token
                vision_attn_mask = F.pad(rpb, (0, 0, 1, 0), mode="constant", value=0.0)

            hidden_states = layer(
                hidden_states,
                query_pos=query_pos,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_mask=text_attn_mask,
                vision_cross_attn_mask=vision_attn_mask,
            )

            # Query updates
            queries = hidden_states[:, 1:]
            normed_queries = self.output_layer_norm(queries)
            intermediate_outputs.append(normed_queries)

            delta = self.box_head(normed_queries)
            ref_inv = inverse_sigmoid(reference_boxes)
            new_reference_boxes = (delta + ref_inv).sigmoid()
            reference_boxes = new_reference_boxes.detach()
            intermediate_boxes.append(new_reference_boxes)

            # Presence score prediction
            presence = self.presence_head(self.presence_layer_norm(hidden_states[:, :1])).squeeze(-1)
            intermediate_presence.append(presence.clamp(min=-10.0, max=10.0))

        return (
            torch.stack(intermediate_outputs),       # [Layers, B, Q, C]
            torch.stack(intermediate_boxes[:-1]),     # [Layers, B, Q, 4]
            torch.stack(intermediate_presence),       # [Layers, B, 1]
        )

# ============================================================================
# Dot-Product Scoring Head
# ============================================================================

class Sam3DotProductScoring(nn.Module):
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.text_mlp = Sam3DecoderMLP(hidden_size, intermediate_size, hidden_size, num_layers=2)
        self.text_mlp_dropout = nn.Dropout(dropout)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size)

        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = 1.0 / np.sqrt(hidden_size)

    def _pool_text(self, text_features: torch.Tensor, text_mask: torch.Tensor | None) -> torch.Tensor:
        if text_mask is None:
            return text_features.mean(dim=1)
        valid = text_mask.to(text_features.dtype).unsqueeze(-1)
        return (text_features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.text_mlp_out_norm(text_features + self.text_mlp_dropout(self.text_mlp(text_features)))
        pooled = self._pool_text(h, text_mask)

        proj_t = self.text_proj(pooled).unsqueeze(-1)         # [B, C, 1]
        proj_q = self.query_proj(decoder_hidden_states)       # [Layers, B, Q, C]

        scores = torch.matmul(proj_q, proj_t.unsqueeze(0)) * self.scale
        return scores.clamp(min=-12.0, max=12.0).squeeze(-1)