import os
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Tuple

from SubModels.Ministral3 import Ministral3Config, Ministral3ForCausalLM
from SubModels.Pixtral import PixtralConfig, PixtralVisionModel
from SubModels.Mistral3MultiModalProjector import Mistral3MultiModalProjector
from Zoo.Mistral3.utils.KVCache import KVCache


# ============================================================
# Configuration
# ============================================================

@dataclass
class Mistral3Config:
    """
    Multimodal configuration that composes:
        - text_config (Ministral3)
        - vision_config (Pixtral)
    """

    spatial_merge_size: int = 2
    image_token_index: int = 10

    text_config: Ministral3Config = field(default_factory=Ministral3Config)
    vision_config: PixtralConfig = field(default_factory=PixtralConfig)


# ============================================================
# Torch compile–safe check helper
# ============================================================

def torch_compilable_check(
    cond: Union[bool, torch.Tensor, Callable[[], Union[bool, torch.Tensor]]],
    msg: Union[str, Callable[[], str]],
    error_type: type[Exception] = ValueError,
) -> None:
    """
    TorchDynamo-compatible check utility.
    Works with torch.compile.
    """

    if os.getenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "0") == "1":
        return

    if not callable(msg):
        def msg_callable():
            return msg
    else:
        msg_callable = msg

    if callable(cond):
        cond = cond()

    if isinstance(cond, torch.Tensor):
        torch._check_tensor_all_with(error_type, cond, msg_callable)
    else:
        torch._check_with(error_type, cond, msg_callable)


# ============================================================
# Main Multimodal Model
# ============================================================

class Mistral3Model(nn.Module):
    """
    Multimodal Mistral3 model.

    Architecture:
        image -> PixtralVisionModel
              -> MultiModalProjector
              -> replace [IMG] token embeddings
              -> Ministral3ForCausalLM
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_tower = PixtralVisionModel(config.vision_config)

        # Vision → text projector
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        # Text model
        self.language_model = Ministral3ForCausalLM(config.text_config)

    # --------------------------------------------------------
    # Embedding helpers
    # --------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    # --------------------------------------------------------
    # Vision processing
    # --------------------------------------------------------

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Runs the vision encoder and projects features into text space.

        Returns:
            Tensor of shape:
                (total_image_tokens, hidden_size)
        """

        vision_outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )

        # Select which hidden layer(s) to use
        if vision_feature_layer is None:
            selected_features = vision_outputs.hidden_states[-1]
        elif isinstance(vision_feature_layer, int):
            selected_features = vision_outputs.hidden_states[vision_feature_layer]
        else:
            selected_features = torch.cat(
                [vision_outputs.hidden_states[i] for i in vision_feature_layer],
                dim=-1,
            )

        # Remove batch dim if present
        if selected_features.dim() == 3 and selected_features.size(0) == 1:
            selected_features = selected_features.squeeze(0)

        # Project into text hidden space
        projected_features = self.multi_modal_projector(
            selected_features,
            image_sizes,
        )

        return projected_features

    # --------------------------------------------------------
    # Placeholder replacement
    # --------------------------------------------------------

    def _replace_image_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replaces image placeholder tokens with projected image features.
        """

        image_token_mask = input_ids == self.config.image_token_index

        num_placeholders = image_token_mask.sum().item()
        num_features = image_features.size(0)

        torch_compilable_check(
            num_placeholders == num_features,
            f"Image tokens ({num_placeholders}) "
            f"!= image features ({num_features})",
        )

        expanded_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds)

        image_features = image_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        return inputs_embeds.masked_scatter(expanded_mask, image_features)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Returns:
            dict with:
                - last_hidden_state
                - past_key_values
                - hidden_states
                - attentions
                - image_hidden_states
        """

        # Validate text input
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must provide input_ids or inputs_embeds.")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of input_ids or inputs_embeds.")

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states: Optional[torch.Tensor] = None

        # Inject image features if provided
        if pixel_values is not None:
            if image_sizes is None:
                raise ValueError("image_sizes must be provided when pixel_values is used.")

            image_hidden_states = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
            )

            # inputs_embeds is guaranteed to be non-None at this point
            assert inputs_embeds is not None
            # input_ids is also guaranteed to be non-None if inputs_embeds was None at the start
            assert input_ids is not None

            inputs_embeds = self._replace_image_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_hidden_states,
            )

        # Run language model
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        return {
            "last_hidden_state": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "image_hidden_states": image_hidden_states,
        }
