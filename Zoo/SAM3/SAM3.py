"""Master SAM3 Architecture integrating detection, multi-scale necks, and video tracking."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from Zoo.Common.vision_utils import (
    box_cxcywh_to_xyxy,
    inverse_sigmoid,
)
from Zoo.SAM3.configs import SAM3Config
from Zoo.SAM3.modules.SAM3DETR import (
    Sam3DetrDecoder,
    Sam3DetrEncoder,
    Sam3DotProductScoring,
)
from Zoo.SAM3.modules.SAM3GeometryEncoder import Sam3GeometryEncoder
from Zoo.SAM3.modules.SAM3MaskDecoder import Sam3MaskDecoder
from Zoo.SAM3.modules.SAM3TextModel import Sam3TextEncoder
from Zoo.SAM3.modules.SAM3TrackerModel import Sam3TrackerModel
from Zoo.SAM3.modules.SAM3VisionModel import Sam3VisionModel, Sam3VisionNeck


class Sam3DetectorModel(nn.Module):
    """Submodule matching all `detector_model.*` checkpoint keys."""

    def __init__(self, cfg: Optional[SAM3Config] = None) -> None:
        super().__init__()
        config = cfg or SAM3Config()

        self.vision_encoder = Sam3VisionModel(config.vision_config)
        self.text_encoder = Sam3TextEncoder(config.text_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.text_config.detr_proj_dim)

        self.geometry_encoder = Sam3GeometryEncoder(
            hidden_size=config.detr_config.hidden_size,
            intermediate_size=config.detr_config.intermediate_size,
            num_layers=3,
            num_heads=config.detr_config.num_heads,
        )

        self.detr_encoder = Sam3DetrEncoder(
            hidden_size=config.detr_config.hidden_size,
            intermediate_size=config.detr_config.intermediate_size,
            num_layers=config.detr_config.num_encoder_layers,
            num_heads=config.detr_config.num_heads,
        )
        self.detr_decoder = Sam3DetrDecoder(
            hidden_size=config.detr_config.hidden_size,
            intermediate_size=config.detr_config.intermediate_size,
            num_layers=config.detr_config.num_decoder_layers,
            num_queries=config.detr_config.num_queries,
            num_heads=config.detr_config.num_heads,
        )

        self.dot_product_scoring = Sam3DotProductScoring(
            hidden_size=config.detr_config.hidden_size,
            intermediate_size=config.detr_config.intermediate_size,
        )
        self.mask_decoder = Sam3MaskDecoder(
            hidden_size=config.detr_config.hidden_size,
            num_heads=config.detr_config.num_heads,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        input_boxes_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # 1. Vision Features (ViT + FPN)
        fpn_features, fpn_positions = self.vision_encoder(pixel_values)
        fpn_features_detr = fpn_features[:-1]
        fpn_positions_detr = fpn_positions[:-1]

        # 2. Text Features
        text_seq = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_seq)
        text_mask = attention_mask.bool() if attention_mask is not None else None

        # 3. Geometric Box Prompts
        if input_boxes is not None and input_boxes.numel() > 0:
            if input_boxes_labels is None:
                input_boxes_labels = torch.ones_like(input_boxes[..., 0], dtype=torch.long)
            box_mask = torch.ones((batch_size, input_boxes.shape[1]), dtype=torch.bool, device=device)

            geo_features, geo_mask = self.geometry_encoder(
                box_embeddings=input_boxes,
                box_mask=box_mask,
                box_labels=input_boxes_labels,
                img_feats=fpn_features_detr,
                img_pos_embeds=fpn_positions_detr,
            )
            combined_prompts = torch.cat([text_features, geo_features], dim=1)
            combined_mask = torch.cat([text_mask, geo_mask], dim=1) if text_mask is not None else geo_mask
        else:
            combined_prompts = text_features
            combined_mask = text_mask

        # 4. Multimodal DETR Encoder
        enc_hidden, enc_pos, _, spatial_shapes = self.detr_encoder(
            vision_features=[fpn_features_detr[-1]],
            text_features=combined_prompts,
            vision_pos_embeds=[fpn_positions_detr[-1]],
            text_mask=combined_mask,
        )

        # 5. DETR Decoder
        inter_outputs, inter_boxes, inter_presence = self.detr_decoder(
            vision_features=enc_hidden,
            text_features=combined_prompts,
            vision_pos_encoding=enc_pos,
            text_mask=combined_mask,
            spatial_shapes=spatial_shapes,
        )

        # 6. Coordinate Refinement & Scoring
        delta_boxes = self.detr_decoder.box_head(inter_outputs)
        pred_boxes_cxcywh = (inverse_sigmoid(inter_boxes) + delta_boxes).sigmoid()
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes_cxcywh[-1])

        pred_logits = self.dot_product_scoring(
            decoder_hidden_states=inter_outputs,
            text_features=combined_prompts,
            text_mask=combined_mask,
        )[-1]

        presence_logits = inter_presence[-1]
        last_queries = inter_outputs[-1]

        # 7. Dense Mask Generation
        pred_masks, semantic_seg = self.mask_decoder(
            decoder_queries=last_queries,
            backbone_features=list(fpn_features_detr),
            encoder_hidden_states=enc_hidden,
            prompt_features=combined_prompts,
            prompt_mask=combined_mask,
        )

        return {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": semantic_seg,
        }


class Sam3Model(nn.Module):
    """Unified SAM3 Architecture matching official checkpoints 1:1.

    Hierarchy:
      - `detector_model.*`: Open-vocabulary detector and segmenter.
      - `tracker_neck.*`: Multi-scale neck for temporal tracking.
      - `tracker_model.*`: SAM2/SAM3 temporal memory and attention engine.
    """

    def __init__(self, config: Optional[SAM3Config] = None) -> None:
        super().__init__()
        self.config = config or SAM3Config()

        self.detector_model = Sam3DetectorModel(self.config)
        self.tracker_neck = Sam3VisionNeck(
            in_channels=self.config.vision_config.hidden_size,
            fpn_dim=self.config.vision_config.fpn_dim,
        )
        self.tracker_model = Sam3TrackerModel(hidden_size=self.config.tracker_config.hidden_size)

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Routes default invocation directly to the detector model."""
        return self.detector_model(*args, **kwargs)