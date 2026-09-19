"""PaliGemma 2 Multimodal Generation Model."""

from typing import Optional
import torch
from torch import nn

from configs import PaliGemma2Config
from Zoo.PaliGemma2.modules.SigLip import SigLipVisionModel
from Zoo.PaliGemma2.modules.Gemma2 import Gemma2ForCausalLM
from Zoo.Common.KV_Cache import KVCache


class PaliGemmaMultiModalProjector(nn.Module):
    """Explicitly typed projector that preserves 'multi_modal_projector.linear.*' keys."""
    def __init__(self, vision_dim: int, text_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(vision_dim, text_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PaliGemma2ForConditionalGeneration(nn.Module):
    def __init__(self, config: Optional[PaliGemma2Config] = None) -> None:
        super().__init__()
        self.config = config or PaliGemma2Config()

        # 1. Vision Tower (SigLip)
        self.vision_tower = SigLipVisionModel(self.config.vision_config)

        # 2. Projector
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_dim=self.config.vision_config.hidden_size,
            text_dim=self.config.projection_dim,
        )

        # 3. Language Model (Gemma 2)
        self.language_model = Gemma2ForCausalLM(self.config.text_config)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_cache: Optional[KVCache] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        b, seq_len = input_ids.shape
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Prefill visual tokens
        if pixel_values is not None:
            vis_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            projected = self.multi_modal_projector.linear(vis_features)
            
            # Normalization scale required by Gemma
            projected = projected / (self.config.text_config.hidden_size ** 0.5)

            # Scatter visual embeddings where input_ids == image_token_index
            mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(mask, projected.view(-1, inputs_embeds.shape[-1]))

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Unified position IDs and causal mask calculation
        cache_len = kv_cache.num_items() if kv_cache else 0
        total_len = cache_len + seq_len

        # Causal mask (0.0 for attend, -inf for masked)
        causal_mask = torch.zeros(b, 1, seq_len, total_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if seq_len > 1:
            causal_triu = torch.triu(torch.full((seq_len, total_len), float("-inf"), device=inputs_embeds.device), diagonal=cache_len + 1)
            causal_mask = causal_mask + causal_triu

        pos_ids = attention_mask.cumsum(-1)[:, -seq_len:]

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=pos_ids,
            past_key_values=kv_cache,
        )