"""Full CLIP architecture aligning dual image-text representations in a shared metric space."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from Zoo.CLIP.configs import CLIPConfig
from Zoo.CLIP.modules.CLIPTextModel import CLIPTextTransformer
from Zoo.CLIP.modules.CLIPVisionModel import CLIPVisionTransformer


class CLIPModel(nn.Module):
    """PyTorch implementation of OpenAI CLIP (Contrastive Language-Image Pre-Training)."""

    def __init__(self, config: Optional[CLIPConfig] = None) -> None:
        super().__init__()
        self.config = config or CLIPConfig()

        # 1. Dual Transformer Towers
        self.text_model = CLIPTextTransformer(self.config.text_config)
        self.vision_model = CLIPVisionTransformer(self.config.vision_config)

        # 2. Multimodal Projections (512-dim default space)
        self.visual_projection = nn.Linear(
            self.config.vision_config.hidden_size,
            self.config.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.config.text_config.hidden_size,
            self.config.projection_dim,
            bias=False,
        )

        # 3. Learnable logit scale temperature
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    def get_text_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encodes and projects text sequences into the shared multimodal space.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).

        Returns:
            Normalized feature tensor of shape (batch_size, projection_dim).
        """
        text_pooled = self.text_model(input_ids, return_pooled=True)
        text_embeds = self.text_projection(text_pooled)
        return text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encodes and projects images into the shared multimodal space.

        Args:
            pixel_values: FloatTensor of shape (batch_size, 3, H, W).

        Returns:
            Normalized feature tensor of shape (batch_size, projection_dim).
        """
        vision_pooled = self.vision_model(pixel_values)
        image_embeds = self.visual_projection(vision_pooled)
        return image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates bidirectional cosine similarity logits between images and text prompts.

        Args:
            input_ids: LongTensor of shape (batch_size_text, seq_len).
            pixel_values: FloatTensor of shape (batch_size_images, 3, H, W).

        Returns:
            Dictionary containing 'logits_per_image' and 'logits_per_text'.
        """
        # 1. Forward passes & L2 Normalization
        image_embeds = self.get_image_features(pixel_values)
        text_embeds = self.get_text_features(input_ids)

        # 2. Scaled Cosine Similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()

        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }