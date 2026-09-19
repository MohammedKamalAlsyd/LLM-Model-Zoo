"""Lightweight, optimized PaliGemma 2 processor."""

from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

from configs import PaliGemma2ProcessorConfig


class PaliGemma2Processor:
    def __init__(self, tokenizer, config: Optional[PaliGemma2ProcessorConfig] = None) -> None:
        self.config = config or PaliGemma2ProcessorConfig()
        self.tokenizer = tokenizer
        
        # Precompute mean and std tensors of shape (3, 1, 1) once for fast broadcasting
        self.mean = torch.tensor(self.config.image_mean, dtype=torch.bfloat16).view(3, 1, 1)
        self.std = torch.tensor(self.config.image_std, dtype=torch.bfloat16).view(3, 1, 1)

        # Add multimodal special tokens if not already present
        tokens_to_add = [self.config.image_token]
        tokens_to_add += [f"<loc{i:04d}>" for i in range(self.config.num_location_tokens)]
        tokens_to_add += [f"<seg{i:03d}>" for i in range(self.config.num_segmentation_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.config.image_token)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Resizes (Bicubic: 3), rescales, and normalizes using config-defined mean and std."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        # Resample=3 is BICUBIC from preprocessor_config.json
        resized = image.convert("RGB").resize(
            (self.config.image_size, self.config.image_size),
            resample=Image.Resampling.BICUBIC
        )

        # (H, W, C) -> (C, H, W) in bfloat16
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).to(torch.bfloat16)

        # 1. Rescale (typically * 1/255.0 to [0, 1])
        tensor = tensor * self.config.rescale_factor

        # 2. Dynamic normalize using configured mean and std
        tensor = (tensor - self.mean) / self.std

        return tensor

    def __call__(
        self,
        text: Union[List[str], str],
        image: Optional[Image.Image] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, list):
            text = text[0]

        pixel_values = None
        if image is not None:
            # (1, 3, H, W)
            pixel_values = self.process_image(image).unsqueeze(0)
            # PaliGemma prefix format: <image>*256<bos>prompt\n
            prompt = f"{self.config.image_token * self.config.image_seq_length}{self.tokenizer.bos_token}{text}\n"
        else:
            prompt = text

        batch = self.tokenizer(prompt, return_tensors=return_tensors)
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        return batch