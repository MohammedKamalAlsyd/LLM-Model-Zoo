# Mistral3_impl.py
"""
Multimodal Mistral-3 style wrapper using your SubModels:
  - SubModels.Pixtral.PixtralVisionModel
  - SubModels.Mistral3MultiModalProjector.Mistral3MultiModalProjector
  - SubModels.Ministral3.Ministral3ForCausalLM

Design choices:
- Uses `torch.Tensor` for typing (Pylance-friendly).
- Forward returns Python dicts with expected keys:
    "last_hidden_state", "past_key_values", "hidden_states", "attentions", "image_hidden_states"
- Image features are projected and inserted into `inputs_embeds` via masked_scatter.
- `prepare_inputs_for_generation` ensures pixel_values are only forwarded in the first generation step.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

import torch
from torch import nn

# Adjust these imports to your repo layout if needed
from SubModels.Ministral3 import Ministral3Config, Ministral3ForCausalLM
from SubModels.Pixtral import PixtralConfig, PixtralVisionModel
from SubModels.Mistral3MultiModalProjector import Mistral3MultiModalProjector
from Zoo.Mistral3.utils.KVCache import KVCache


@dataclass
class Mistral3Config:
    """
    High-level multimodal configuration composition.
    """
    spatial_merge_size: int = 2
    image_token_index: int = 10
    text_config: Ministral3Config = field(default_factory=Ministral3Config)
    vision_config: PixtralConfig = field(default_factory=PixtralConfig)


def torch_compilable_check(cond, msg, error_type: type = ValueError) -> None:
    """
    Minimal compile-safe check wrapper. Mirrors earlier helper but simplified.
    It's safe to leave as runtime checks — the important part for Pylance is the
    type narrowing via asserts in forward.
    """
    if not cond:
        raise error_type(msg)


class Mistral3Model(nn.Module):
    """
    Multimodal model that composes:
      PixtralVisionModel -> Mistral3MultiModalProjector -> Ministral3ForCausalLM

    The model expects that the tokenizer prompt contains placeholder tokens whose id
    equals `config.image_token_index`. The number of placeholders must match the number
    of merged image tokens produced by the projector.
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_tower = PixtralVisionModel(config.vision_config)

        # Projector: vision -> text hidden space
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        # Text model (causal LM)
        self.language_model = Ministral3ForCausalLM(config.text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(value)

    def _vision_forward_and_project(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Run vision tower and projector.

        Returns:
            projected_features: (total_merged_patches, text_hidden_size)
            The function concatenates merged patches for all images in batch order.
        """
        # call vision tower; your PixtralTransformer returns a dict-like (or tuple),
        # but the PixtralVisionModel implementation you gave returns a dict with
        # "last_hidden_state" (batch dim may be 1). Adjust as necessary.
        vision_outputs = self.vision_tower(
            pixel_values, image_sizes=image_sizes, output_hidden_states=True, return_dict=True
        )

        # Determine the selected features (use last hidden state if unspecified)
        # We accept the case where vision_outputs["last_hidden_state"] has shape (1, total_patches, v_hidden)
        # or (batch, seq, v_hidden). For compatibility with your earlier code, we handle the leading 1.
        # The PixtralVisionModel in your repo flattens per-image patches and returns shape (1, total_patches, hidden)
        hv = vision_outputs["last_hidden_state"]
        if vision_feature_layer is not None:
            # If the vision model returns full hidden states per layer, handle that case:
            hs = vision_outputs.get("hidden_states", None)
            if hs is not None:
                if isinstance(vision_feature_layer, int):
                    hv = hs[vision_feature_layer]
                else:
                    hv = torch.cat([hs[i] for i in vision_feature_layer], dim=-1)

        # remove leading batch dim if present and equals 1
        if hv.dim() == 3 and hv.size(0) == 1:
            hv = hv.squeeze(0)  # (total_patches, v_hidden)

        # project to text space
        projected = self.multi_modal_projector(hv, image_sizes)

        # projected is (total_merged_patches, text_hidden)
        return projected

    def _replace_image_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace image placeholder token embeddings in `inputs_embeds` with `image_features`.

        - input_ids: (batch, seq_len)
        - inputs_embeds: (batch, seq_len, text_hidden)
        - image_features: (total_image_tokens, text_hidden)

        Returns:
            new_inputs_embeds: same shape as inputs_embeds with placeholders replaced.
        """
        # Ensure shapes and non-None via asserts (helps Pylance)
        assert input_ids is not None, "input_ids required to replace image tokens"
        assert inputs_embeds is not None, "inputs_embeds required to replace image tokens"
        assert image_features is not None, "image_features required to replace image tokens"

        # Build a boolean mask for placeholder tokens; mask shape (batch, seq_len)
        image_token_mask = input_ids == self.config.image_token_index  # dtype=bool

        num_placeholders = int(image_token_mask.sum().item())
        num_image_features = int(image_features.size(0))

        torch_compilable_check(
            num_placeholders == num_image_features,
            f"Number of image placeholders ({num_placeholders}) does not match number of image features ({num_image_features})",
        )

        # Expand mask to embeddings shape and scatter
        expanded_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds)  # (batch, seq_len, hidden)
        image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # masked_scatter expects the source to be flattened in row-major order where mask is True.
        new_embeds = inputs_embeds.masked_scatter(expanded_mask, image_features)

        return new_embeds

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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Multimodal forward.

        Accepts either `input_ids` or `inputs_embeds` (but not both). If `pixel_values` is provided,
        `image_sizes` must also be provided.

        Returns:
            dict containing:
              - last_hidden_state: (batch, seq_len, hidden)
              - past_key_values: KVCache or None
              - hidden_states: optional tuple/list or None
              - attentions: optional tuple/list or None
              - image_hidden_states: projected image features (total_merged_patches, hidden) or None
        """
        # Validate boundary inputs (explicit and Pylance-friendly)
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of input_ids or inputs_embeds")

        # If no embeddings provided, create them from input_ids
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Keep image_hidden_states for return
        image_hidden_states: Optional[torch.Tensor] = None

        # If pixel_values provided, compute image features and replace placeholders
        if pixel_values is not None:
            if image_sizes is None:
                raise ValueError("image_sizes must be provided when passing pixel_values")

            # Project vision -> text
            image_hidden_states = self._vision_forward_and_project(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
            )

            # Ensure tensors are well-typed for Pylance
            assert inputs_embeds is not None
            assert input_ids is not None  # input_ids must exist because we need to locate placeholders

            # Replace placeholder tokens with image features
            inputs_embeds = self._replace_image_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_hidden_states,
            )

        # Run language model (Ministral3ForCausalLM) using inputs_embeds
        outputs = self.language_model(
            input_ids=None,  # we pass inputs_embeds directly
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        # `Ministral3ForCausalLM` in your SubModels returns a dict: {"logits", "past_key_values"}.
        # But your underlying model may also return additional info depending on your implementation.
        # We'll handle both dict and object-like returns gracefully.

        # If the language model returns dict-like:
        if isinstance(outputs, dict):
            last_hidden = outputs.get("last_hidden_state", None) or outputs.get("logits", None)
            past_kv = outputs.get("past_key_values", None)
            hidden_states = outputs.get("hidden_states", None)
            attentions = outputs.get("attentions", None)
        else:
            # In case your Ministral3ForCausalLM returns an object with attributes
            last_hidden = getattr(outputs, "last_hidden_state", None) or getattr(outputs, "logits", None)
            past_kv = getattr(outputs, "past_key_values", None)
            hidden_states = getattr(outputs, "hidden_states", None)
            attentions = getattr(outputs, "attentions", None)

        return {
            "last_hidden_state": last_hidden,
            "past_key_values": past_kv,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "image_hidden_states": image_hidden_states,
        }


class Mistral3ForConditionalGeneration(nn.Module):
    """
    Wrapper for multimodal generation:
      - contains Mistral3Model
      - lm_head for logits over vocabulary (weight tied to token embeddings)
      - forward that computes logits and optional loss
      - prepare_inputs_for_generation to handle pixel_values only at first iteration
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config
        self.model = Mistral3Model(config)
        text_hidden = config.text_config.hidden_size
        vocab_size = config.text_config.vocab_size

        self.lm_head = nn.Linear(text_hidden, vocab_size, bias=False)
        # Tie weights to token embeddings if possible
        try:
            self.lm_head.weight = self.model.get_input_embeddings().weight
        except Exception:
            # If tie fails, leave lm_head initialized normally
            pass

        # Loss function used when labels provided
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forward for conditional generation.

        Returns dict with keys:
          - logits: (batch, seq_len, vocab_size)
          - loss: optional scalar tensor when labels provided
          - past_key_values: optional KVCache
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        # The language model inside returns last_hidden_state (hidden states), which we need to project to logits.
        last_hidden = outputs["last_hidden_state"]
        if last_hidden is None:
            raise RuntimeError("Language model forward did not return last_hidden_state")

        # Optionally slice last N tokens for decoder speed-up
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_hidden[:, slice_indices, :])

        loss = None
        if labels is not None:
            # align shapes: (batch*seq_len, vocab_size) vs (batch*seq_len,)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"logits": logits, "loss": loss, "past_key_values": outputs.get("past_key_values", None)}

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
        """
        Prepare model inputs for generation loops.

        Only include pixel_values in the very first iteration (or when use_cache=False),
        because image tokens are merged into the model inputs and then stored in the KV cache.
        """
        model_inputs: Dict[str, Any] = {
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
