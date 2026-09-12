import torch
import torch.nn as nn
from Zoo.SAM3.SubModels.SAM3VisionModel import Sam3VisionModel
from Zoo.SAM3.SubModels.SAM3TextModel import Sam3TextEncoder
from Zoo.SAM3.SubModels.SAM3GeometryEncoder import Sam3GeometryEncoder
from Zoo.SAM3.SubModels.SAM3DETR import Sam3DetrEncoder, Sam3DetrDecoder, Sam3DotProductScoring
from Zoo.SAM3.SubModels.SAM3MaskDecoder import Sam3MaskDecoder
from Zoo.SAM3.SubModels.SAM3Common import box_cxcywh_to_xyxy, inverse_sigmoid

class Sam3Model(nn.Module):
    """
    SAM3 Model Architecture.
    All attribute names and sub-module hierarchies are configured to match
    the official checkpoint keys 1:1, allowing direct strict=True loading.
    """
    def __init__(self):
        super().__init__()
        # 1. Vision Backbone (ViT + Multi-scale FPN Neck)
        self.vision_encoder = Sam3VisionModel()

        # 2. Text Backbone (CLIP Text Encoder)
        self.text_encoder = Sam3TextEncoder(hidden_size=1024, projection_dim=512)

        # 3. Text Dimension Projection: 1024 -> 256
        self.text_projection = nn.Linear(1024, 256)

        # 4. Geometry & Prompt Encoder
        self.geometry_encoder = Sam3GeometryEncoder(hidden_size=256, intermediate_size=2048, num_layers=3, num_heads=8)

        # 5. DETR Encoder & Decoder
        self.detr_encoder = Sam3DetrEncoder(hidden_size=256, intermediate_size=2048, num_layers=6, num_heads=8)
        self.detr_decoder = Sam3DetrDecoder(hidden_size=256, intermediate_size=2048, num_layers=6, num_queries=200, num_heads=8)

        # 6. Dot Product Query Scoring
        self.dot_product_scoring = Sam3DotProductScoring(hidden_size=256, intermediate_size=2048)

        # 7. Mask Decoder
        self.mask_decoder = Sam3MaskDecoder(hidden_size=256, num_heads=8)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_boxes: torch.Tensor | None = None,
        input_boxes_labels: torch.Tensor | None = None,
    ):
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # 1. Vision Features
        fpn_features, fpn_positions = self.vision_encoder(pixel_values)
        fpn_features_detr = fpn_features[:-1]
        fpn_positions_detr = fpn_positions[:-1]

        # 2. Text Features
        text_seq = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_seq)
        text_mask = attention_mask.bool() if attention_mask is not None else None

        # 3. Geometric Prompts (if any)
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
            # Concatenate text & geometric prompts
            combined_prompts = torch.cat([text_features, geo_features], dim=1)
            combined_mask = torch.cat([text_mask, geo_mask], dim=1) if text_mask is not None else geo_mask
        else:
            combined_prompts = text_features
            combined_mask = text_mask

        # 4. DETR Encoder
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

        # 6. Box Refinement & Scoring
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

        # 7. Mask Generation
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