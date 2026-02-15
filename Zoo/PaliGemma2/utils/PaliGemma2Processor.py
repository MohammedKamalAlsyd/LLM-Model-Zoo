"""PaliGemma2 Processor Module.

This module extends the Gemma2 tokenizer to handle special tokens for image processing,
object detection (bounding boxes), and segmentation tasks. It provides a unified interface
for tokenizing text and processing images in a multi-modal setting.

Classes:
    PaliGemma2Processor: Main processor class that handles text tokenization and image
        preprocessing for the PaliGemma2 model.
"""

from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch import nn

__all__ = ["PaliGemma2Processor"]

# ImageNet normalization constants
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class PaliGemma2Processor(nn.Module):
    """Multi-modal processor for PaliGemma2 model.

    Extends Gemma2 tokenizer with special tokens for:
    - Image tokens: <image> for image placeholder
    - Location tokens: <loc0000> to <loc1023> for object detection bounding boxes
    - Segmentation tokens: <seg000> to <seg127> for object segmentation masks

    Attributes:
        IMAGE_TOKEN (str): Special token representing image placeholder.
        image_tokens (int): Number of image tokens / patches in vision encoder output.
        image_size (int): Expected input image size (square images).
        rescale_factor (float): Factor to rescale pixel values (default: 0.5).
        image_token_id (int): Token ID of IMAGE_TOKEN after tokenizer update.
    """

    IMAGE_TOKEN = "<image>"
    NUM_LOCATION_TOKENS = 1024
    NUM_SEGMENTATION_TOKENS = 128

    def __init__(
        self,
        tokenizer,
        image_tokens: int,
        image_size: int,
        rescale_factor: float = (1.0 / 255.0),
    ) -> None:
        """Initialize the PaliGemma2Processor.

        Args:
            tokenizer: Tokenizer instance (e.g., from transformers library).
            image_tokens (int): Number of image patch tokens from vision encoder.
            image_size (int): Input image size (assumed square).
            rescale_factor (float): Pixel rescaling factor. Defaults to 0.5.

        Raises:
            ValueError: If image_tokens or image_size are invalid.
        """
        super().__init__()

        if image_tokens <= 0:
            raise ValueError(f"image_tokens must be positive, got {image_tokens}")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")

        self.image_tokens = image_tokens
        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.tokenizer = tokenizer

        # Add image token as special token
        tokens_to_add = {"img_token": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # Add location tokens for bounding box coordinates (object detection)
        location_tokens = [f"<loc{i:04d}>" for i in range(self.NUM_LOCATION_TOKENS)]

        # Add segmentation tokens for mask indices (semantic/instance segmentation)
        segmentation_tokens = [
            f"<seg{i:03d}>" for i in range(self.NUM_SEGMENTATION_TOKENS)
        ]

        # Add both token sets to tokenizer
        tokenizer.add_tokens(location_tokens + segmentation_tokens)

        # Store image token ID for later use
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # Manage BOS/EOS tokens manually
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def process_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """Process image into normalized tensor format.

        Steps:
            1. Convert to RGB if necessary
            2. Resize to target image_size
            3. Rescale pixel values by rescale_factor
            4. Normalize using ImageNet statistics
            5. Convert to tensor and rearrange to (C, H, W) format

        Args:
            image (Image.Image): Input PIL image in any format.

        Returns:
            torch.Tensor: Processed image tensor of shape (3, image_size, image_size)
                with values normalized to [-1, 1] range (for standard ImageNet stats).

        Raises:
            TypeError: If image is not a PIL Image instance.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to target dimensions (square image)
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )

        # Convert PIL image to numpy array (H, W, C) with values in [0, 1]
        image_array = torch.tensor(image, dtype=torch.bfloat16) / 255.0

        # Rescale pixel values
        image_array = image_array * self.rescale_factor

        # Rearrange to (C, H, W) format
        image_array = image_array.permute(2, 0, 1)

        # Normalize using ImageNet statistics
        mean = torch.tensor(IMAGENET_STANDARD_MEAN, dtype=torch.bfloat16).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STANDARD_STD, dtype=torch.bfloat16).view(3, 1, 1)
        image_tensor = (image_array - mean) / std

        return image_tensor

    def __call__(
        self,
        text: Union[List[str], str],
        image: Optional[Image.Image] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process text and image inputs for the PaliGemma2 model.

        Formats inputs as: [BOS] <image> * image_seq_len [prefix_prompt]

        Args:
            text (Union[List[str], str]): Text prompt(s). If list, processes first element.
            image (Optional[Image.Image]): Input image to process. If None, only text is processed.
            return_tensors (str): Format of returned tensors. Defaults to "pt" (PyTorch).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'input_ids': Token IDs of shape (1, seq_len)
                - 'attention_mask': Attention mask of shape (1, seq_len)
                - 'pixel_values': Processed image tensor of shape (1, 3, image_size, image_size)
                    (only if image is provided)

        Raises:
            TypeError: If text is not str or list of str.
            ValueError: If text list is empty.
        """
        # Normalize text input
        if isinstance(text, list):
            if not text:
                raise ValueError("Text list cannot be empty")
            text = text[0]
        elif not isinstance(text, str):
            raise TypeError(f"text must be str or List[str], got {type(text)}")

        # Process image if provided
        pixel_values = None
        if image is not None:
            pixel_values = self.process_image(image)
            pixel_values = pixel_values.unsqueeze(
                0
            )  # Add batch dimension: (1, 3, H, W)

        # Build prompt with image tokens
        if pixel_values is not None:
            image_tokens_str = self.IMAGE_TOKEN * self.image_tokens
            prompt = f"{image_tokens_str}{self.tokenizer.bos_token}{text}\n"
        else:
            prompt = text

        # Tokenize the prompt using the extended tokenizer
        input = self.tokenizer(
            prompt,
            return_tensors=return_tensors,
        )

        output = {**input}

        if pixel_values is not None:
            output["pixel_values"] = pixel_values

        return output
