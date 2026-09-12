import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer
from Zoo.SAM3.SubModels.SAM3Common import box_xyxy_to_cxcywh

class SAM3Processor:
    """
    Self-contained processor for SAM3 handling image resizing, normalization,
    text tokenization, bounding box normalization, and post-processing.
    """
    def __init__(self, tokenizer_id: str = "openai/clip-vit-base-patch32", target_size: int = 1008):
        self.target_size = target_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            max_length=32,
            model_max_length=32
        )

        # Standard ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def process_image(self, image: Image.Image, device: str = "cpu") -> tuple[torch.Tensor, tuple[int, int]]:
        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")

        # Native PIL resize guarantees PIL.Image.Image type for TF.to_tensor
        img_resized = img_rgb.resize(
            (self.target_size, self.target_size),
            resample=Image.Resampling.BILINEAR
        )
        tensor_img = TF.to_tensor(img_resized)
        norm_img = TF.normalize(tensor_img, mean=self.mean, std=self.std)

        pixel_values = norm_img.unsqueeze(0).to(device)
        return pixel_values, (orig_h, orig_w)

    def process_text(self, text: str | list[str], device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def process_boxes(
        self,
        boxes: list[list[float]] | torch.Tensor,
        orig_size: tuple[int, int],
        device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes raw pixel bounding boxes in [x1, y1, x2, y2] format, normalizes to [0, 1],
        and transforms to [cx, cy, w, h] format.
        """
        orig_h, orig_w = orig_size
        if not isinstance(boxes, torch.Tensor):
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_t = boxes.clone().float()

        if boxes_t.ndim == 2:
            boxes_t = boxes_t.unsqueeze(0)  # [1, N, 4]

        # Normalize to [0, 1]
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
        boxes_norm = boxes_t / scale
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_norm).to(device)

        labels = torch.ones((boxes_cxcywh.shape[0], boxes_cxcywh.shape[1]), dtype=torch.long, device=device)
        return boxes_cxcywh, labels

    @staticmethod
    def post_process_masks(
        pred_masks: torch.Tensor,
        orig_size: tuple[int, int],
        threshold: float = 0.0
    ) -> torch.Tensor:
        """Interpolates mask logits back to original image size and binarizes."""
        orig_h, orig_w = orig_size
        upscaled = F.interpolate(
            pred_masks,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        return (upscaled > threshold).cpu()

    @staticmethod
    def post_process_detections(
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        presence_logits: torch.Tensor | None,
        orig_size: tuple[int, int],
        score_threshold: float = 0.3
    ) -> list[dict[str, torch.Tensor]]:
        """Scales relative boxes [0, 1] to original image coordinates."""
        orig_h, orig_w = orig_size
        scores = pred_logits.sigmoid()
        if presence_logits is not None:
            scores = scores * presence_logits.sigmoid()

        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=pred_boxes.device, dtype=pred_boxes.dtype)
        boxes_scaled = pred_boxes * scale

        results = []
        for b_scores, b_boxes in zip(scores, boxes_scaled):
            keep = b_scores > score_threshold
            results.append({
                "scores": b_scores[keep].cpu(),
                "boxes": b_boxes[keep].cpu()
            })
        return results