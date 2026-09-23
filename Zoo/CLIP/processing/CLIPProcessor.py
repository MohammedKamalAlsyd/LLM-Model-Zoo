"""Universal image preprocessing and prompt tokenization pipeline for CLIP."""

from typing import Dict, List, Optional, Union, Any
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

from Zoo.CLIP.configs import CLIPProcessorConfig


class CLIPProcessor:
    """End-to-end multimodal input pipeline matching OpenAI CLIP specifications."""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        config: Optional[CLIPProcessorConfig] = None,
        tokenizer_id: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.config = config or CLIPProcessorConfig()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_id)

        # Precompute mean and standard deviation tensors for fast broadcasting
        self.mean = torch.tensor(self.config.image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(self.config.image_std, dtype=torch.float32).view(3, 1, 1)

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Executes RGB conversion, aspect-preserving bicubic resize, center-crop, and normalization.

        Args:
            image: Source PIL Image.

        Returns:
            FloatTensor of shape (3, H, W).
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        # 1. Color mode conversion
        if self.config.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Aspect-preserving resize (short edge to config.image_size)
        w, h = image.size
        target = self.config.image_size
        if self.config.do_resize:
            if h < w:
                new_h = target
                new_w = int(w * (target / h))
            else:
                new_w = target
                new_h = int(h * (target / w))
            image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        # 3. Center crop to (image_size, image_size)
        if self.config.do_center_crop:
            w, h = image.size
            left = (w - target) // 2
            top = (h - target) // 2
            image = image.crop((left, top, left + target, top + target))

        # 4. Convert to FloatTensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        # 5. Rescale and Normalize
        if self.config.do_rescale:
            tensor = tensor * self.config.rescale_factor
        if self.config.do_normalize:
            tensor = (tensor - self.mean) / self.std

        return tensor

    def process_text(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Encodes candidate text categories using BPE tokenization."""
        if isinstance(text, str):
            text = [text]

        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors=return_tensors,
        )

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Unified callable preparing both image and text inputs in a single batch dictionary."""
        batch = {}

        if text is not None:
            batch.update(self.process_text(text, return_tensors=return_tensors))

        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            batch["pixel_values"] = torch.stack([self.process_image(img) for img in images])

        return batch