"""Universal Visual Grounding Studio for Open-Vocabulary Segmentation & Video Tracking.

Enables zero-shot spatial concept localization on images and temporal propagation
across video sequences without conversational dialogue overhead.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

# Setup repository paths
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent if (current_dir.parent / "Zoo").exists() else current_dir
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights, purge_memory
from Zoo.SAM3.SAM3 import Sam3Model
from Zoo.SAM3.processing.SAM3Processor import SAM3Processor

DEFAULT_GROUNDING_REPO = "facebook/sam3"
DEVICE, DTYPE = auto_detect_device_and_dtype()

PALETTE = [
    [255, 60, 60],
    [60, 255, 60],
    [60, 120, 255],
    [255, 215, 0],
    [255, 60, 255],
    [0, 255, 255],
    [255, 140, 0],
]

# --- Predefined Grounding Task Examples ---
IMAGE_EXAMPLES = [
    [
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800",
        "dog",
        0.35,
    ],
    [
        "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=800",
        "sports car",
        0.30,
    ],
    [
        "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=800",
        "yellow bandana",
        0.25,
    ],
    [
        "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800",
        "vintage car wheel",
        0.30,
    ],
]

VIDEO_EXAMPLES = [
    [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "car",
        0.30,
        16,
    ],
    [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
        "bicycle",
        0.30,
        16,
    ],
    [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/bottle-detection.mp4",
        "bottle",
        0.35,
        16,
    ],
]


def download_if_url(path_or_url: Optional[str]) -> Optional[str]:
    """Downloads remote images/videos with request headers preventing 403 Forbidden errors."""
    if not path_or_url or not isinstance(path_or_url, str):
        return None
    if path_or_url.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "modelzoo_cache")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(path_or_url.split("?")[0])
        local_filepath = os.path.join(cache_dir, filename)

        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) > 0:
            return local_filepath

        try:
            req = urllib.request.Request(
                path_or_url,
                headers={"User-Agent": "LLMModelZoo/1.0 Python-urllib"},
            )
            with urllib.request.urlopen(req, timeout=20) as response, open(local_filepath, "wb") as out_file:
                out_file.write(response.read())
            return local_filepath
        except Exception as e:
            print(f"Warning: Failed to fetch remote resource {path_or_url}: {e}")
            return None
    return path_or_url


def resolve_image(image_input: Any) -> Optional[Image.Image]:
    """Resolves arbitrary Gradio inputs (PIL, file paths, URLs, NumPy arrays) into a RGB PIL Image."""
    if image_input is None:
        return None
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        resolved_path = download_if_url(image_input)
        if resolved_path and os.path.exists(resolved_path):
            return Image.open(resolved_path).convert("RGB")
        return None
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    return None


class VisualGroundingManager:
    """Manages memory lifecycle and lazy-loading for grounding backbones."""

    def __init__(self) -> None:
        self.model: Optional[Sam3Model] = None
        self.processor: Optional[SAM3Processor] = None
        self.active_repo: Optional[str] = None

    def unload(self) -> None:
        """Purges active model weights from system RAM and GPU VRAM."""
        if self.model is not None:
            print(f"[*] Unloading '{self.active_repo}'...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.active_repo = None
            purge_memory()

    def get_model(self, repo_id: str = DEFAULT_GROUNDING_REPO) -> Tuple[Sam3Model, SAM3Processor]:
        """Lazy loads weights with shard streaming and strict verification."""
        if self.model is not None and self.processor is not None and self.active_repo == repo_id:
            return self.model, self.processor

        self.unload()
        print(f"[*] Initializing Visual Grounding Engine on {DEVICE.upper()} in {DTYPE}...")
        model = Sam3Model()

        load_hf_model_weights(
            model=model,
            repo_id=repo_id,
            allow_patterns=["*.safetensors", "*.bin"],
            strict=True,
            device=DEVICE,
            dtype=DTYPE,
        )
        processor = SAM3Processor()

        self.model = model
        self.processor = processor
        self.active_repo = repo_id
        return model, processor


MANAGER = VisualGroundingManager()


def draw_overlay(image: Image.Image, masks: torch.Tensor) -> np.ndarray:
    """Applies semi-transparent high-contrast masks onto a PIL image."""
    img_np = np.array(image.convert("RGB")).astype(np.float32)

    for i in range(min(len(masks), 25)):
        mask = masks[i].numpy()
        if not mask.any():
            continue
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        img_np[mask] = img_np[mask] * 0.45 + color * 0.55

    return np.clip(img_np, 0, 255).astype(np.uint8)


@torch.no_grad()
def ground_image(
    image_input: Any,
    text_prompt: str,
    threshold: float,
) -> Tuple[Optional[np.ndarray], str]:
    """Single-frame open-vocabulary concept localization."""
    image = resolve_image(image_input)
    if image is None:
        return None, "Please upload or select an input image."
    prompt = text_prompt.strip() or "object"

    model, processor = MANAGER.get_model()
    model_dtype = next(model.parameters()).dtype

    pixel_values, orig_size = processor.process_image(image, device=DEVICE)
    pixel_values = pixel_values.to(dtype=model_dtype)
    input_ids, attention_mask = processor.process_text(prompt, device=DEVICE)

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    raw_masks = outputs["pred_masks"][0]
    scores = outputs["pred_logits"][0].sigmoid()
    if outputs["presence_logits"] is not None:
        scores = scores * outputs["presence_logits"][0].sigmoid()

    keep = scores > threshold
    filtered_masks = raw_masks[keep]

    if filtered_masks.shape[0] == 0:
        return np.array(image), f"No instances found matching '{prompt}' (threshold: {threshold:.2f})."

    binary_masks = processor.post_process_masks(filtered_masks.unsqueeze(0), orig_size=orig_size)[0]
    annotated = draw_overlay(image, binary_masks)
    return annotated, f"Successfully grounded {filtered_masks.shape[0]} instance(s) of '{prompt}'."


@torch.no_grad()
def ground_video(
    video_input: Optional[str],
    text_prompt: str,
    threshold: float,
    max_frames: int = 16,
) -> Tuple[Optional[str], str]:
    """Temporal grounding and video object tracking across sampled frames."""
    if not video_input:
        return None, "Please upload or select a valid video."

    resolved_video_path = download_if_url(video_input)
    if not resolved_video_path or not os.path.exists(resolved_video_path):
        return None, "Could not resolve video source file."

    prompt = text_prompt.strip() or "object"
    model, processor = MANAGER.get_model()
    model_dtype = next(model.parameters()).dtype

    batched_pixels, pil_frames, orig_size, _ = processor.process_video(
        video_path=resolved_video_path,
        max_frames=int(max_frames),
        device=DEVICE,
    )
    batched_pixels = batched_pixels.to(dtype=model_dtype)
    input_ids, attention_mask = processor.process_text(prompt, device=DEVICE)

    # 1. Ground initial anchor frame
    init_outputs = model(
        pixel_values=batched_pixels[:1],
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    scores = init_outputs["pred_logits"][0].sigmoid()
    if init_outputs["presence_logits"] is not None:
        scores = scores * init_outputs["presence_logits"][0].sigmoid()

    keep = scores > threshold
    active_masks = init_outputs["pred_masks"][0][keep]

    annotated_frames: List[np.ndarray] = []

    # 2. Sequential frame tracking
    for idx, frame in enumerate(pil_frames):
        if idx == 0 and active_masks.shape[0] > 0:
            b_masks = processor.post_process_masks(active_masks.unsqueeze(0), orig_size=orig_size)[0]
            annotated_frames.append(draw_overlay(frame, b_masks))
        else:
            out = model(
                pixel_values=batched_pixels[idx : idx + 1],
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            f_scores = out["pred_logits"][0].sigmoid()
            if out["presence_logits"] is not None:
                f_scores = f_scores * out["presence_logits"][0].sigmoid()
            f_keep = f_scores > threshold
            f_masks = out["pred_masks"][0][f_keep]

            if f_masks.shape[0] > 0:
                b_masks = processor.post_process_masks(f_masks.unsqueeze(0), orig_size=orig_size)[0]
                annotated_frames.append(draw_overlay(frame, b_masks))
            else:
                annotated_frames.append(np.array(frame.convert("RGB")))

    # 3. Write tracked output video
    temp_dir = tempfile.gettempdir()
    out_video_path = os.path.join(temp_dir, "grounding_tracked_output.mp4")

    h, w = orig_size
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter.fourcc(*"mp4v"), 8.0, (w, h))
    for f in annotated_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    return out_video_path, f"Tracked '{prompt}' across {len(annotated_frames)} video frames."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Visual Grounding Studio") as app:
        gr.Markdown("# 🎯 Visual Grounding & Pixel Segmentation Studio")
        gr.Markdown(
            f"Zero-Shot Spatial Localization & Video Tracking running on **`{DEVICE.upper()}`** in **`{DTYPE}`**.<br/>"
            "Grounds open-vocabulary natural language concepts into pixel-accurate segmentation masks."
        )

        with gr.Tabs():
            # -------------------------------------------------------------
            # Tab 1: Static Image Grounding
            # -------------------------------------------------------------
            with gr.TabItem("🖼️ Static Image Grounding"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="Input Image")
                        img_prompt = gr.Textbox(label="Concept Prompt", value="dog")
                        img_thresh = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Detection Confidence")
                        img_btn = gr.Button("Ground Concept", variant="primary")

                    with gr.Column():
                        img_output = gr.Image(type="numpy", label="Segmented Output")
                        img_status = gr.Textbox(label="Grounding Summary")

                gr.Markdown("### 💡 Predefined Image Examples")
                gr.Examples(
                    examples=IMAGE_EXAMPLES,
                    inputs=[img_input, img_prompt, img_thresh],
                    outputs=[img_output, img_status],
                    fn=ground_image,
                    cache_examples=False,
                )

                img_btn.click(
                    fn=ground_image,
                    inputs=[img_input, img_prompt, img_thresh],
                    outputs=[img_output, img_status],
                )

            # -------------------------------------------------------------
            # Tab 2: Spatio-Temporal Video Grounding
            # -------------------------------------------------------------
            with gr.TabItem("🎥 Spatio-Temporal Video Grounding"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Source Video")
                        vid_prompt = gr.Textbox(label="Target Concept to Track", value="car")
                        vid_thresh = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="Tracking Confidence")
                        vid_frames = gr.Slider(4, 32, value=16, step=2, label="Sampled Frame Count")
                        vid_btn = gr.Button("Track Over Video Sequence", variant="primary")

                    with gr.Column():
                        vid_output = gr.Video(label="Tracked Output Video")
                        vid_status = gr.Textbox(label="Tracking Status")

                gr.Markdown("### 💡 Predefined Video Examples")
                gr.Examples(
                    examples=VIDEO_EXAMPLES,
                    inputs=[vid_input, vid_prompt, vid_thresh, vid_frames],
                    outputs=[vid_output, vid_status],
                    fn=ground_video,
                    cache_examples=False,
                )

                vid_btn.click(
                    fn=ground_video,
                    inputs=[vid_input, vid_prompt, vid_thresh, vid_frames],
                    outputs=[vid_output, vid_status],
                )

    return app


if __name__ == "__main__":
    server = build_ui()
    server.launch(server_name="0.0.0.0", server_port=7862, share=True)