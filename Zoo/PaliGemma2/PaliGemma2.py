"""
PaliGemma2 Multimodal Configuration

Composes:
- Gemma2 (text model)
- SigLip (vision model)

This file wires together the submodule configs using the
values provided in the reference JSON.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from torch import nn
import torch

# Import sub-configs directly from your structure
from SubModels.Gemma2 import Gemma2Config, Gemma2ForCausalLM
from SubModels.SigLip import SigLipVisionConfig, SigLipVisionModel
from SubModels.PaliGemmaProjector import PaliGemmaMultiModalProjector
from utils.KVCache import KVCache


class PaliGemma2Config:
    """
    Top-level configuration for the PaliGemma2 multimodal model.

    Combines:
        - Gemma2 text configuration
        - SigLip vision configuration
        - Cross-modal projection settings
    """

    # -------- Sub-model configs --------
    text_config: Gemma2Config
    vision_config: SigLipVisionConfig

    # -------- Multimodal --------
    model_type: str = "paligemma"
    image_token_index: int = 257152
    projection_dim: int = 2304
    torch_dtype: str = "bfloat16"

    def __init__(self, main_config: dict[str, Any]) -> None:
        """
        Initialize sub-configs using values from the provided JSON
        if they were not explicitly passed.
        """

        self.config = main_config  # Store the main config for reference

        # Access the needed values from the main config
        self.bos_token_id = main_config.get("bos_token_id", 2)
        self.eos_token_id = main_config.get("eos_token_id", 1)
        self.pad_token_id = main_config.get("pad_token_id", 0)

        # Sub-model configs
        text_config_dict = main_config.get("text_config", {})
        vision_config_dict = main_config.get("vision_config", {})

        # Ensure text_config_dict is a dictionary before unpacking
        keys_to_keep = Gemma2Config.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in text_config_dict.items() if k in keys_to_keep}
        filtered_dict["pad_token_id"] = self.pad_token_id
        filtered_dict["eos_token_id"] = self.eos_token_id
        filtered_dict["bos_token_id"] = self.bos_token_id
        self.text_config = Gemma2Config(**filtered_dict)

        # Ensure vision_config_dict is a dictionary before unpacking
        keys_to_keep = SigLipVisionConfig.__dataclass_fields__.keys()
        filtered_dict = {
            k: v for k, v in vision_config_dict.items() if k in keys_to_keep
        }
        self.vision_config = SigLipVisionConfig(**filtered_dict)


class PaliGemma2ForConditionalGeneration(nn.Module):
    """
    PaliGemma2 multimodal model for conditional generation.

    Combines:
        - Gemma2 text model
        - SigLip vision model
        - Cross-modal projection layers
    """

    def __init__(self, config: PaliGemma2Config):
        """
        Initialize the PaliGemma2 model with the given configuration.

        Args:
            config (PaliGemma2Config): Model configuration.
        """
        super().__init__()
        self.config = config  # Store config for use in forward pass

        # Initialize sub-models
        self.language_model = Gemma2ForCausalLM(config.text_config)
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            hidden_size=config.projection_dim,
            vision_projection_dim=config.vision_config.projection_dim,
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,  # (B, N_img, H)
        inputs_embeds: torch.Tensor,  # (B, T, H)
        input_ids: torch.Tensor,  # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
        kv_cache: Optional[KVCache] = None,  # kept generic on purpose
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace image token embeddings with projected vision features
        and build causal attention mask + position ids.
        """

        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # --------------------------------------------------
        # Scale image features (Gemma-style)
        # --------------------------------------------------
        image_features = image_features / (hidden_size**0.5)

        # Final embedding buffer
        final_embeddings = torch.zeros(
            batch_size, seq_len, hidden_size, device=device, dtype=dtype
        )

        # Token masks
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Expand masks to embedding dim
        text_mask = torch.tensor(text_mask, device=device).unsqueeze(-1)
        image_mask = torch.tensor(image_mask, device=device).unsqueeze(-1)
        pad_mask = torch.tensor(pad_mask, device=device).unsqueeze(-1)

        # Insert text embeddings
        final_embeddings = torch.where(text_mask, inputs_embeds, final_embeddings)

        # Insert image embeddings (sequence-aligned scatter)
        final_embeddings = final_embeddings.masked_scatter(
            image_mask.expand_as(final_embeddings),
            image_features.reshape(-1),
        )

        # Zero out padding
        final_embeddings = torch.where(
            pad_mask, torch.zeros_like(final_embeddings), final_embeddings
        )

        # --------------------------------------------------
        # Build causal attention mask
        # --------------------------------------------------
        q_len = seq_len

        if kv_cache is None:
            # Prefill phase (no cache)
            causal_mask = torch.zeros(
                batch_size, q_len, q_len, device=device, dtype=dtype
            )
        else:
            # Decode phase (single token query)
            assert q_len == 1
            kv_len = kv_cache.num_items() + 1
            causal_mask = torch.zeros(batch_size, 1, kv_len, device=device, dtype=dtype)

        # Add head dimension: (B, 1, Q, K)
        causal_mask = causal_mask.unsqueeze(1)

        # --------------------------------------------------
        # Position IDs (Gemma2 expects these)
        # --------------------------------------------------
        if kv_cache is not None:
            # Last token position only
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # Full prefill positions
            position_ids = attention_mask.cumsum(-1)
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)

        return final_embeddings, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> dict:
        """
        Forward pass for multimodal conditional generation.
        """

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        # This implementation assumes NO padding (same as PaliGemma-1)
        assert torch.all(attention_mask == 1), "Padding is not supported"

        # --------------------------------------------------
        # Text embeddings
        # --------------------------------------------------
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # --------------------------------------------------
        # Vision tower
        # --------------------------------------------------
        vision_outputs = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features = self.multi_modal_projector(vision_outputs)

        # --------------------------------------------------
        # Merge modalities
        # --------------------------------------------------
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        )

        # --------------------------------------------------
        # Language model forward
        # --------------------------------------------------
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
        )

        return outputs
