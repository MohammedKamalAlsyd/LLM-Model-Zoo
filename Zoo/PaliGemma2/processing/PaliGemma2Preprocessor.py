"""Preprocessing module for PaliGemma 2 input images and text prompts."""

from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

from configs import PaliGemma2ProcessorConfig


class PaliGemma2Preprocessor:
    """Handles image transformation pipeline and multimodal prompt framing."""

    def __init__(self, tokenizer, config: Optional[PaliGemma2ProcessorConfig] = None) -> None:
        """Initializes preprocessor with tokenizer and configuration.

        Args:
            tokenizer: Hugging Face tokenizer instance (e.g. AutoTokenizer).
            config: PaliGemma2ProcessorConfig containing preprocessing hyperparams.
        """
        self.config = config or PaliGemma2ProcessorConfig()
        self.tokenizer = tokenizer

        # Precompute mean and standard deviation tensors for broadcasting
        self.mean = torch.tensor(self.config.image_mean, dtype=torch.bfloat16).view(3, 1, 1)
        self.std = torch.tensor(self.config.image_std, dtype=torch.bfloat16).view(3, 1, 1)

        # Register multimodal special tokens if missing
        tokens_to_add = [self.config.image_token]
        tokens_to_add += [f"<loc{i:04d}>" for i in range(self.config.num_location_tokens)]
        tokens_to_add += [f"<seg{i:03d}>" for i in range(self.config.num_segmentation_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.config.image_token)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Applies configured RGB conversion, resize, rescale, and normalization.

        Args:
            image: Source PIL Image.

        Returns:
            torch.Tensor of shape (3, H, W) in bfloat16.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        # 1. Color mode conversion
        if self.config.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Resize
        if self.config.do_resize:
            image = image.resize(
                (self.config.image_size, self.config.image_size),
                resample=Image.Resampling(self.config.resample),
            )

        # (H, W, C) -> (C, H, W) in bfloat16
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.bfloat16)

        # 3. Rescale
        if self.config.do_rescale:
            tensor = tensor * self.config.rescale_factor

        # 4. Normalize
        if self.config.do_normalize:
            tensor = (tensor - self.mean) / self.std

        return tensor

    def __call__(
        self,
        text: Union[List[str], str],
        image: Optional[Image.Image] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Prepares multimodal batch with repeated image tokens and prompt.

        Args:
            text: Text prompt string or single-item list of strings.
            image: Optional PIL Image input.
            return_tensors: Tensor format for tokenizer output (default: 'pt').

        Returns:
            Dictionary containing 'input_ids', 'attention_mask', and optionally 'pixel_values'.
        """
        if isinstance(text, list):
            text = text[0]

        pixel_values = None
        if image is not None:
            pixel_values = self.process_image(image).unsqueeze(0)
            # PaliGemma standard prefix: 256 <image> tokens + <bos> + prompt + newline
            prompt = f"{self.config.image_token * self.config.image_seq_length}{self.tokenizer.bos_token}{text}\n"
        else:
            prompt = text

        batch = self.tokenizer(prompt, return_tensors=return_tensors)
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        return batch