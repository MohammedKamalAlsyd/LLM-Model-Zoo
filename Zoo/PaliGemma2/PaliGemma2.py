"""PaliGemma 2 Multimodal Conditional Generation Model."""

from typing import Optional
import torch
from torch import nn

from configs import PaliGemma2Config
from Zoo.PaliGemma2.modules.SigLip import SigLipVisionModel
from Zoo.PaliGemma2.modules.Gemma2 import Gemma2ForCausalLM
from Zoo.Common.KV_Cache import KVCache


class PaliGemmaMultiModalProjector(nn.Module):
    """Linear projector aligning vision representation dimension with language embedding space."""

    def __init__(self, vision_dim: int, text_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(vision_dim, text_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PaliGemma2ForConditionalGeneration(nn.Module):
    """Full PaliGemma 2 architecture unifying SigLIP and Gemma 2."""

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
        """Ties LM head weights with word embeddings."""
        self.language_model.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_cache: Optional[KVCache] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass for multimodal prefill and autoregressive decode.

        Args:
            input_ids: (Batch, Seq_Len) token IDs.
            kv_cache: KVCache instance for caching past key/values.
            pixel_values: Optional (Batch, 3, H, W) normalized image tensor.
            attention_mask: Optional 2D binary attention mask (1 for valid, 0 for pad).

        Returns:
            Dict containing output logits: {"logits": (Batch, Seq_Len, Vocab_Size)}.
        """
        b, seq_len = input_ids.shape
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # 1. Fuse Image Features (Matches official masked_scatter)
        if pixel_values is not None:
            vis_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            projected = self.multi_modal_projector(vis_features)

            mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(mask, projected.view(-1, inputs_embeds.shape[-1]))

        # Cache lengths and total sequence length
        cache_len = kv_cache.num_items() if kv_cache is not None else 0
        total_len = cache_len + seq_len

        # Position IDs start at 1, NOT 0:
        pos_ids = (torch.arange(cache_len, total_len, device=input_ids.device, dtype=torch.long) + 1).unsqueeze(0).expand(b, -1)
        causal_mask = torch.zeros(b, 1, seq_len, total_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if attention_mask is not None and attention_mask.shape[-1] == total_len:
            pad_mask = (attention_mask == 0).view(b, 1, 1, total_len)
            causal_mask = causal_mask.masked_fill(pad_mask, float("-inf"))

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids,
            attention_mask=causal_mask,
            past_key_values=kv_cache,
        )