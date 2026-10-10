"""Multimodal Image, Video, and Prompt Preprocessing Pipeline for SAM3."""

from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer

from Zoo.Common.multimodality_utils import sample_video_frames
from Zoo.Common.vision_utils import box_xyxy_to_cxcywh
from Zoo.SAM3.configs import SAM3ProcessorConfig


class SAM3Processor:
    """Unified multimodal processor for images, temporal video frames, and geometric prompts."""

    def __init__(
        self,
        config: Optional[SAM3ProcessorConfig] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_id: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.config = config or SAM3ProcessorConfig()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_id,
            max_length=self.config.max_text_length,
            model_max_length=self.config.max_text_length,
        )

    def process_image(
        self,
        image: Image.Image,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocesses a single PIL Image with high-quality bilinear resizing and normalization.

        Args:
            image: Source PIL image.
            device: Target execution device.

        Returns:
            pixel_values of shape (1, 3, target_size, target_size) and original (height, width).
        """
        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")

        # Native PIL bilinear resize guarantees high fidelity
        img_resized = img_rgb.resize(
            (self.config.target_size, self.config.target_size),
            resample=Image.Resampling.BILINEAR,
        )
        tensor_img = TF.to_tensor(img_resized)
        norm_img = TF.normalize(tensor_img, mean=list(self.config.image_mean), std=list(self.config.image_std))

        pixel_values = norm_img.unsqueeze(0).to(device)
        return pixel_values, (orig_h, orig_w)

    def process_video(
        self,
        video_path: str,
        max_frames: int = 16,
        sample_fps: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, List[Image.Image], Tuple[int, int], List[float]]:
        """Extracts and standardizes video frames for temporal tracking.

        Args:
            video_path: Path to MP4/AVI/MKV video file.
            max_frames: Upper bound on extracted frames.
            sample_fps: Target frame rate sampling.
            device: Target execution device.

        Returns:
            Tuple of:
                - batched_pixels: Tensor of shape (num_frames, 3, target_size, target_size).
                - pil_frames: List of raw PIL Images for downstream rendering.
                - orig_size: (height, width) of the raw video stream.
                - timestamps: Frame timestamp floats in seconds.
        """
        pil_frames, timestamps = sample_video_frames(video_path, max_frames=max_frames, sample_fps=sample_fps)
        if not pil_frames:
            raise ValueError(f"No frames could be extracted from: {video_path}")

        orig_w, orig_h = pil_frames[0].size
        tensors = []
        for frame in pil_frames:
            img_resized = frame.resize(
                (self.config.target_size, self.config.target_size),
                resample=Image.Resampling.BILINEAR,
            )
            t = TF.to_tensor(img_resized)
            tensors.append(TF.normalize(t, mean=list(self.config.image_mean), std=list(self.config.image_std)))

        batched_pixels = torch.stack(tensors, dim=0).to(device)
        return batched_pixels, pil_frames, (orig_h, orig_w), timestamps

    def process_text(
        self,
        text: Union[str, List[str]],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text prompts for open-vocabulary detection."""
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.config.max_text_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def process_boxes(
        self,
        boxes: Union[List[List[float]], torch.Tensor],
        orig_size: Tuple[int, int],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes pixel bounding boxes [x1, y1, x2, y2] to center format [cx, cy, w, h]."""
        orig_h, orig_w = orig_size
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if not isinstance(boxes, torch.Tensor) else boxes.clone().float()

        if boxes_t.ndim == 2:
            boxes_t = boxes_t.unsqueeze(0)

        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
        boxes_norm = boxes_t / scale
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_norm).to(device)
        labels = torch.ones((boxes_cxcywh.shape[0], boxes_cxcywh.shape[1]), dtype=torch.long, device=device)
        return boxes_cxcywh, labels

    @staticmethod
    def post_process_masks(
        pred_masks: torch.Tensor,
        orig_size: Tuple[int, int],
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Interpolates mask logits back to native image dimensions and binarizes."""
        orig_h, orig_w = orig_size
        upscaled = F.interpolate(pred_masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return (upscaled > threshold).cpu()

    @staticmethod
    def post_process_detections(
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        presence_logits: Optional[torch.Tensor],
        orig_size: Tuple[int, int],
        score_threshold: float = 0.35,
    ) -> List[Dict[str, torch.Tensor]]:
        """Scales normalized [0, 1] bounding box coordinates back to absolute pixel values."""
        orig_h, orig_w = orig_size
        scores = pred_logits.sigmoid()
        if presence_logits is not None:
            scores = scores * presence_logits.sigmoid()

        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=pred_boxes.device, dtype=pred_boxes.dtype)
        boxes_scaled = pred_boxes * scale

        results: List[Dict[str, torch.Tensor]] = []
        for b_scores, b_boxes in zip(scores, boxes_scaled):
            keep = b_scores > score_threshold
            results.append({
                "scores": b_scores[keep].cpu(),
                "boxes": b_boxes[keep].cpu(),
            })
        return results