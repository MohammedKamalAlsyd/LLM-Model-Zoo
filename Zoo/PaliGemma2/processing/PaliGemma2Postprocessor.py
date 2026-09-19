"""Postprocessing module for PaliGemma 2 with localization and segmentation coordinate parsing."""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch


class PaliGemma2Postprocessor:
    """Extracts bounding box coordinates, segmentation IDs, and renders visual annotations."""

    def __init__(self, tokenizer=None, mask_decoder=None) -> None:
        """Initializes postprocessor.

        Args:
            tokenizer: Optional tokenizer for raw token-id decoding.
            mask_decoder: Optional UViM MaskDecoder for rendering instance segmentation masks.
        """
        self.tokenizer = tokenizer
        self.mask_decoder = mask_decoder
        self.loc_pattern = re.compile(r"<loc(\d{4})>")
        self.seg_pattern = re.compile(r"<seg(\d{3})>")
        # Matches: <loc####><loc####><loc####><loc####> [optional <seg###> tokens] [optional label text]
        self.detection_pattern = re.compile(r"((?:<loc\d{4}>){4})\s*((?:<seg\d{3}>)+)?\s*([^<;\n]+)?")
        # Matches: sequence of <seg###> tokens [optional label text]
        self.segment_pattern = re.compile(r"((?:<seg\d{3}>)+)\s*([^<;\n]+)?")

    def parse_locations(self, text: str, width: int, height: int) -> List[Dict[str, Any]]:
        """Parses [ymin, xmin, ymax, xmax] coordinates from generated text.

        PaliGemma encodes coordinates via 4 location tokens: <locY1><locX1><locY2><locX2>
        where each integer is normalized to [0, 1024].

        Args:
            text: Raw generated string containing <loc####> tokens.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            List of dicts: [{"box_2d": [ymin, xmin, ymax, xmax], "label": str}, ...]
        """
        detections = []
        matches = self.detection_pattern.finditer(text)

        for match in matches:
            raw_locs = self.loc_pattern.findall(match.group(1))
            if len(raw_locs) < 4:
                continue

            # Normalized [0, 1024] -> Absolute Pixel coordinates
            y1 = int(int(raw_locs[0]) / 1024.0 * height)
            x1 = int(int(raw_locs[1]) / 1024.0 * width)
            y2 = int(int(raw_locs[2]) / 1024.0 * height)
            x2 = int(int(raw_locs[3]) / 1024.0 * width)

            ymin, ymax = max(0, min(y1, y2)), min(height, max(y1, y2))
            xmin, xmax = max(0, min(x1, x2)), min(width, max(x1, x2))

            label = match.group(3).strip() if match.group(3) else ""
            seg_tokens = self.seg_pattern.findall(match.group(2)) if match.group(2) else []
            segment_ids = [int(idx) for idx in seg_tokens] if len(seg_tokens) == 16 else None

            detections.append({
                "box_2d": [ymin, xmin, ymax, xmax],
                "label": label,
                "segment_ids": segment_ids,
            })

        return detections

    def parse_segments(self, text: str) -> List[Dict[str, Any]]:
        """Extracts segmentation token sequences (<seg###>) and associated entity labels.

        Args:
            text: Raw generated string containing <seg###> tokens.

        Returns:
            List of dicts: [{"segment_ids": List[int], "label": str}, ...]
        """
        segments = []
        matches = self.segment_pattern.finditer(text)

        for match in matches:
            seg_tokens = self.seg_pattern.findall(match.group(1))
            indices = [int(idx) for idx in seg_tokens]
            label = match.group(2).strip() if match.group(2) else ""
            segments.append({
                "segment_ids": indices,
                "label": label,
            })

        return segments

    def render_bounding_boxes(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        outline_color: str = "#00FF00",
        width: int = 3,
    ) -> Image.Image:
        """Renders bounding rectangles and labels onto a copy of the source PIL image.

        Args:
            image: Source PIL Image.
            detections: List of detection dictionaries containing 'box_2d' and 'label'.
            outline_color: Hex or named color for rectangle border.
            width: Line width for bounding box outline.

        Returns:
            Annotated PIL Image.
        """
        annotated = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(annotated)

        for det in detections:
            ymin, xmin, ymax, xmax = det["box_2d"]
            label = det["label"]
            segment_ids = det.get("segment_ids")

            # Decode and composite segmentation mask if decoder and 16 seg tokens are present
            if segment_ids is not None and self.mask_decoder is not None:
                box_w, box_h = max(1, xmax - xmin), max(1, ymax - ymin)
                tokens_tensor = torch.tensor([segment_ids], dtype=torch.long, device=next(self.mask_decoder.parameters()).device)
                with torch.no_grad():
                    mask_64 = self.mask_decoder(tokens_tensor)[0].cpu().numpy()

                mask_pil = Image.fromarray((mask_64 >= 0.5).astype(np.uint8) * 255, mode="L").resize(
                    (box_w, box_h), resample=Image.Resampling.BILINEAR
                )
                mask_crop = (np.array(mask_pil) > 128).astype(np.uint8)

                color = (0, 255, 255, 120)  # Semi-transparent Cyan
                colored_mask = np.zeros((annotated.height, annotated.width, 4), dtype=np.uint8)
                crop_rgba = np.zeros((box_h, box_w, 4), dtype=np.uint8)
                for c in range(3):
                    crop_rgba[:, :, c] = mask_crop * color[c]
                crop_rgba[:, :, 3] = mask_crop * color[3]
                colored_mask[ymin:ymax, xmin:xmax] = crop_rgba
                overlay = Image.alpha_composite(overlay, Image.fromarray(colored_mask, mode="RGBA"))

            # Draw bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=width)

            # Draw banner with label text
            if label:
                box_top = max(0, ymin - 16)
                draw.rectangle([xmin, box_top, xmin + max(40, len(label) * 9), ymin], fill=outline_color)
                draw.text((xmin + 3, box_top + 1), label, fill="black")

        return Image.alpha_composite(annotated, overlay).convert("RGB")

    def process(
        self,
        generated_ids: List[int],
        image: Optional[Image.Image] = None,
    ) -> Tuple[str, Optional[Image.Image]]:
        """Processes token IDs, decodes text, and renders detection annotations if present.

        Args:
            generated_ids: List of integer token IDs output by the model.
            image: Original PIL Image.

        Returns:
            Tuple of (clean_text, annotated_image_or_None).
        """
        if self.tokenizer is None:
            raise ValueError("A tokenizer must be provided to decode token IDs.")

        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        annotated_image = None

        # Check for localization coordinates
        if "<loc" in raw_text and image is not None:
            detections = self.parse_locations(raw_text, width=image.width, height=image.height)
            if detections:
                annotated_image = self.render_bounding_boxes(image, detections)

        clean_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return clean_text, annotated_image