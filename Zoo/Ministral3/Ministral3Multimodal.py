"""
Ministral-3 8B Multimodal Wrapper (mistralai/Ministral-3-8B-Instruct-2512)

This wrapper connects the unmodified Pixtral vision tower, Mistral3 projector, 
and Ministral3 language model.
"""

from typing import Optional, Union, Dict, Any

import torch
from torch import nn

# Import your untouched submodels
from SubModels.Ministral3 import Ministral3Model
from SubModels.Pixtral import PixtralVisionModel
from SubModels.Mistral3MultiModalProjector import Mistral3MultiModalProjector
from utils.KVCache import KVCache
from config import Ministral3MultimodalConfig


# ============================================================================
# Core Multimodal Model
# ============================================================================

class Ministral3MultimodalModel(nn.Module):
    def __init__(self, config: Ministral3MultimodalConfig):
        super().__init__()
        self.config = config

        # 1. Vision Encoder (using our patched version)
        self.vision_tower = PixtralVisionModel(config.vision_config)

        # 2. Projector
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        # 3. Text Encoder
        self.language_model = Ministral3Model(config.text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.embed_tokens = value

    def _vision_forward_and_project(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[int] = None,
    ) -> torch.Tensor:
        vision_outputs = self.vision_tower(
            pixel_values, image_sizes=image_sizes, output_hidden_states=True
        )

        # Handle specific vision layer extraction
        if vision_feature_layer is None:
            vision_feature_layer = self.config.vision_feature_layer

        if vision_feature_layer == -1:
            hv = vision_outputs["last_hidden_state"]
        else:
            hs = vision_outputs.get("hidden_states", None)
            hv = hs[vision_feature_layer]

        if hv.dim() == 3 and hv.size(0) == 1:
            hv = hv.squeeze(0)

        # Project vision into text space
        projected = self.multi_modal_projector(hv, image_sizes)
        return projected

    def _replace_image_tokens(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Robust masking that works even if input_ids is None"""
        
        if input_ids is None:
            # Fallback: find the image tokens by matching embedding weights
            image_token_tensor = torch.tensor(
                self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device
            )
            expected_image_embed = self.get_input_embeddings()(image_token_tensor)
            image_token_mask = (inputs_embeds == expected_image_embed).all(dim=-1)
        else:
            image_token_mask = (input_ids == self.config.image_token_index)

        num_placeholders = int(image_token_mask.sum().item())
        num_image_features = int(image_features.size(0))

        if num_placeholders != num_image_features:
            raise ValueError(f"Tokens mismatch! Found {num_placeholders} placeholders but {num_image_features} image features.")

        expanded_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).contiguous()

        return inputs_embeds.masked_scatter(expanded_mask, image_features)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None

        if pixel_values is not None:
            if image_sizes is None:
                raise ValueError("image_sizes must be provided when passing pixel_values")

            image_hidden_states = self._vision_forward_and_project(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )

            assert image_hidden_states is not None, "Vision forward failed to produce features"
            assert inputs_embeds is not None, "inputs_embeds should be initialized by this point"

            inputs_embeds = self._replace_image_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_hidden_states,
            )

        # Pass directly to base language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        return {
            "last_hidden_state": outputs["last_hidden_state"],
            "past_key_values": outputs.get("past_key_values", None),
            "image_hidden_states": image_hidden_states,
        }


# ============================================================================
# Conditional Generation Wrapper (The final model)
# ============================================================================

class Ministral3ForConditionalGeneration(nn.Module):
    def __init__(self, config: Ministral3MultimodalConfig):
        super().__init__()
        self.config = config
        self.model = Ministral3MultimodalModel(config)
        
        # Instantiate the LM Head natively
        text_hidden = config.text_config.hidden_size
        vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(text_hidden, vocab_size, bias=False)

        # Tie embeddings based on the JSON config
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            image_sizes=image_sizes,
        )

        last_hidden = outputs["last_hidden_state"]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_hidden[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "logits": logits, 
            "loss": loss, 
            "past_key_values": outputs.get("past_key_values", None)
        }

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        is_first_iteration: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "logits_to_keep": logits_to_keep,
        }

        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs