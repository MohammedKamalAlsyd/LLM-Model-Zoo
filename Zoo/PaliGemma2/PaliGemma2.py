import torch
from torch import nn
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# Sub-models
from SubModels.Gemma2 import Gemma2Config, Gemma2ForCausalLM
from SubModels.SigLip import SigLipVisionConfig, SigLipVisionModel
from SubModels.PaliGemmaProjector import PaliGemmaMultiModalProjector
from utils.KVCache import KVCache

@dataclass
class PaliGemma2Config:
    """
    Hardcoded configuration for 'google/paligemma2-3b-pt-224'.
    Removes the need for external config.json files.
    """
    # Multimodal settings
    model_type: str = "paligemma"
    image_token_index: int = 257152
    torch_dtype: str = "bfloat16"
    
    # Vision (SigLip-So400m)
    vision_config: SigLipVisionConfig = field(default_factory=lambda: SigLipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=27,
        patch_size=14,
        projection_dim=2304, # Output of Vision -> Input to Projector
        image_size=224,
        num_channels=3
    ))

    # Text (Gemma2-2B)
    text_config: Gemma2Config = field(default_factory=lambda: Gemma2Config(
        vocab_size=257216, # Includes 1024 loc + 128 seg tokens
        hidden_size=2304,
        intermediate_size=9216,
        num_attention_heads=8,
        num_key_value_heads=4, # GQA
        num_hidden_layers=26,
        sliding_window=4096,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        query_pre_attn_scalar=2304**-0.5,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        rope_theta=10000
    ))
    
    # Projection
    projection_dim: int = 2304 # Must match text_config.hidden_size

class PaliGemma2ForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemma2Config):
        super().__init__()
        self.config = config or PaliGemma2Config()
        
        # 1. Vision Encoder (SigLip)
        self.vision_tower = SigLipVisionModel(self.config.vision_config)
        
        # 2. Projector (Linear: Vision Dim -> Text Dim)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            hidden_size=self.config.projection_dim,
            vision_projection_dim=self.config.vision_config.hidden_size 
        )
        
        # 3. Text Decoder (Gemma2)
        self.language_model = Gemma2ForCausalLM(self.config.text_config)

    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_inputs(
        self,
        input_ids: torch.LongTensor,      # (B, T)
        inputs_embeds: torch.Tensor,      # (B, T, D_text)
        image_features: torch.Tensor,     # (B, N_patches, D_text)
        attention_mask: torch.Tensor,     # (B, T)
        kv_cache: Optional[KVCache]       # Cache object
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merges text embeddings with projected image features.
        B: Batch, T: Seq Len, D: Hidden Dim
        """
        B, T, D = inputs_embeds.shape
        device = inputs_embeds.device

        # Scale image features (Gemma2 requirement)
        image_features = image_features / (D ** 0.5)

        # 1. Embeddings merging
        # Create a mask where (1) is text, (0) is image placeholder or padding
        # mask shape: (B, T, 1)
        special_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        
        # Scatter image features into the sequence where special_mask is True
        inputs_embeds = inputs_embeds.masked_scatter(special_mask, image_features.view(-1, D))

        # 2. Causal Mask Construction
        # Prefill: (B, 1, T, T), Decode: (B, 1, 1, T_total)
        if kv_cache is None:
            # Full causal mask for prefill
            causal_mask = torch.zeros(B, 1, T, T, device=device, dtype=inputs_embeds.dtype)
        else:
            # Single token decoding
            cache_len = kv_cache.num_items()
            causal_mask = torch.zeros(B, 1, 1, cache_len + 1, device=device, dtype=inputs_embeds.dtype)
            
        # 3. Position IDs
        # Calculate positions based on attention mask (ignoring padding)
        if kv_cache is None:
            position_ids = attention_mask.cumsum(-1).masked_fill(attention_mask == 0, 1)
        else:
            position_ids = attention_mask.cumsum(-1)[:, -1:]

        return inputs_embeds, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,          # (B, T)
        pixel_values: torch.FloatTensor,      # (B, 3, H_img, W_img)
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        # Default mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 1. Text Embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids) # (B, T, D)

        # 2. Vision Embeddings
        # Extract features -> Project to text space
        vis_out = self.vision_tower(pixel_values.to(inputs_embeds.dtype)) # (B, N, D_vis)
        image_features = self.multi_modal_projector(vis_out)              # (B, N, D)

        # 3. Merge Modalities
        final_embeds, causal_mask, pos_ids = self._merge_inputs(
            input_ids, inputs_embeds, image_features, attention_mask, kv_cache
        )

        # 4. Language Model Forward
        return self.language_model(
            inputs_embeds=final_embeds,
            attention_mask=causal_mask,
            position_ids=pos_ids,
            past_key_values=kv_cache
        )