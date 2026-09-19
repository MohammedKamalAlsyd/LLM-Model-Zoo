"""Master Multimodal Wrapper for Ministral-3 / Pixtral.

Unifies Pixtral Vision, Patch Projector, and Ministral-3 Causal Language Model
with 1:1 Hugging Face checkpoint alignment.
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from Zoo.Ministral3.configs import Ministral3MultimodalConfig
from Zoo.Ministral3.modules.Ministral3 import Ministral3ForCausalLM
from Zoo.Ministral3.modules.Mistral3MultiModalProjector import Mistral3MultiModalProjector
from Zoo.Ministral3.modules.PixtralVision import PixtralVisionModel
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.vision_utils import replace_image_tokens


class Mistral3ForConditionalGeneration(nn.Module):
    """Top-level Multimodal conditional generation model for Ministral-3.

    Parameter Hierarchy matches official checkpoints exactly:
      - vision_tower: PixtralVisionModel
      - multi_modal_projector: Mistral3MultiModalProjector
      - language_model: Ministral3ForCausalLM (model + lm_head)
    """

    def __init__(self, config: Ministral3MultimodalConfig) -> None:
        super().__init__()
        self.config = config

        self.vision_tower = PixtralVisionModel(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = Ministral3ForCausalLM(config.text_config)

    def tie_weights(self) -> None:
        """Ties lm_head to token embeddings (invoked automatically by model_loader)."""
        self.language_model.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, slice] = 0,
    ) -> Dict[str, Any]:
        """Conditional generation forward pass.

        During prefill: Pass both `pixel_values` and `input_ids`.
        During autoregressive decode: Pass only `input_ids` with `pixel_values=None` to bypass vision computation.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 1. Process vision and inject into sequence embeddings
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided alongside pixel_values to map image placeholders.")

            # Forward through vision tower: (1, total_patches, vision_hidden)
            raw_vision_features = self.vision_tower(pixel_values, image_sizes=image_sizes)
            # Remove dummy batch dim: (total_patches, vision_hidden)
            if raw_vision_features.ndim == 3 and raw_vision_features.shape[0] == 1:
                raw_vision_features = raw_vision_features.squeeze(0)

            # Project and merge 2x2 blocks: (total_merged_tokens, text_hidden)
            projected_features = self.multi_modal_projector(raw_vision_features, image_sizes=image_sizes)

            # Scatter into embeddings
            inputs_embeds = self.replace_image_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=projected_features,
            )

        # 2. Run language model backbone
        return self.language_model(
            input_ids=None,  # Always pass inputs_embeds directly
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            logits_to_keep=logits_to_keep,
        )