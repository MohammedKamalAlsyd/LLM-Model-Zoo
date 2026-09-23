"""Unified Zero-Shot Studio for Contrastive Language-Image Models (CLIP).

Features dynamic lazy loading, URL image resolution, automatic precision casting,
and interactive probability visualization.
"""

import gc
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image
import torch

# Setup repository path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent if (current_dir.parent / "Zoo").exists() else current_dir
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Zoo.CLIP.CLIP import CLIPModel
from Zoo.CLIP.processing.CLIPProcessor import CLIPProcessor
from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights

# --- Checkpoint Identifier Constants ---
DEFAULT_CLIP_REPO = "openai/clip-vit-base-patch32"

DEVICE, DTYPE = auto_detect_device_and_dtype()

# --- Task Preset Configurations ---
TASK_PRESETS = {
    "General Object Identification": {
        "image": "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=600",
        "labels": "a photo of a dog, a photo of a cat, a picture of outer space, a fast sports car",
    },
    "Landscape & Scene Categorization": {
        "image": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600",
        "labels": "a beautiful landscape, the planet earth from space, a dark room, microscopic cells under a microscope",
    },
    "Vehicle Classification": {
        "image": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=600",
        "labels": "a vintage muscle car, a modern sports coupe, a commuter bicycle, a street motorcycle",
    },
}


def download_if_url(image_path: Optional[str]) -> Optional[str]:
    """Downloads remote images with headers preventing 403 Forbidden responses."""
    if not image_path or not isinstance(image_path, str):
        return None
    if image_path.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "modelzoo_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_filename = os.path.join(cache_dir, os.path.basename(image_path.split("?")[0]))
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
            return local_filename
        try:
            req = urllib.request.Request(
                image_path,
                headers={"User-Agent": "LLMModelZoo/1.0 Python-urllib"},
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(local_filename, "wb") as out_file:
                out_file.write(response.read())
            return local_filename
        except Exception as e:
            print(f"Warning: Could not download image {image_path}: {e}")
            return None
    return image_path


class CLIPManager:
    """Manages VRAM allocation and lifecycle for CLIP models."""

    def __init__(self) -> None:
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.active_repo: Optional[str] = None

    def unload(self) -> None:
        """Purges active model weights from memory and reclaims VRAM."""
        if self.model is not None:
            print(f"[*] Unloading '{self.active_repo}' to free VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.active_repo = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_model(self, repo_id: str = DEFAULT_CLIP_REPO) -> Tuple[CLIPModel, CLIPProcessor]:
        """Loads requested checkpoint on demand if not already resident in memory."""
        if self.model is not None and self.processor is not None and self.active_repo == repo_id:
            return self.model, self.processor

        self.unload()
        print(f"[*] Loading '{repo_id}' onto {DEVICE.upper()} in {DTYPE}...")

        model = CLIPModel()
        load_hf_model_weights(
            model=model,
            repo_id=repo_id,
            allow_patterns=["*.safetensors", "*.bin"],
            strict=True,
            device=DEVICE,
            dtype=DTYPE,
        )
        processor = CLIPProcessor(tokenizer_id=repo_id)

        self.model = model
        self.processor = processor
        self.active_repo = repo_id
        print("✓ CLIP successfully initialized.")
        return model, processor


MANAGER = CLIPManager()


@torch.no_grad()
def run_zero_shot_classification(image_input: Any, labels_text: str) -> Dict[str, float]:
    """Inference endpoint computing normalized image-text alignment probabilities."""
    if image_input is None:
        return {"Error: Please provide an image.": 1.0}

    classes = [c.strip() for c in labels_text.split(",") if c.strip()]
    if not classes:
        return {"Error: Please enter at least one candidate label.": 1.0}

    # Resolve PIL Image from either path, URL, or direct Gradio input
    img: Optional[Image.Image] = None
    if isinstance(image_input, str) and image_input.strip():
        resolved_path = download_if_url(image_input)
        if resolved_path and os.path.exists(resolved_path):
            img = Image.open(resolved_path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")

    if img is None:
        return {"Error: Invalid image source.": 1.0}

    model, processor = MANAGER.get_model()
    model_dtype = next(model.parameters()).dtype

    batch = processor(text=classes, images=img)
    pixel_values = batch["pixel_values"].to(DEVICE, dtype=model_dtype)
    input_ids = batch["input_ids"].to(DEVICE)

    outputs = model(input_ids=input_ids, pixel_values=pixel_values)

    # Cast to float32 before calling .cpu().numpy() to prevent BFloat16 conversion errors
    probs = (
        outputs["logits_per_image"]
        .softmax(dim=-1)
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    sorted_indices = np.argsort(-probs)
    return {classes[idx]: float(probs[idx]) for idx in sorted_indices}


def build_ui() -> gr.Blocks:
    first_preset_name = list(TASK_PRESETS.keys())[0]
    first_preset = TASK_PRESETS[first_preset_name]
    initial_img_path = download_if_url(first_preset["image"])

    with gr.Blocks(title="CLIP Zero-Shot Studio") as app:
        gr.Markdown("# 🔍 Universal CLIP Zero-Shot Studio")
        gr.Markdown(
            f"Dual-Tower Metric Architecture running on **`{DEVICE.upper()}`** in **`{DTYPE}`**.<br/>"
            "Calculates normalized cosine similarity between high-dimensional image and text projections."
        )

        with gr.Row():
            with gr.Column(scale=1):
                preset_selector = gr.Dropdown(
                    choices=list(TASK_PRESETS.keys()),
                    value=first_preset_name,
                    label="Task Preset",
                )

                input_img = gr.Image(
                    type="pil",
                    value=initial_img_path,
                    label="Input Image",
                )

                labels_input = gr.Textbox(
                    label="Candidate Categories (Comma-Separated)",
                    value=first_preset["labels"],
                    lines=3,
                )

                submit_btn = gr.Button("Classify Image", variant="primary")

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Predicted Class Probabilities",
                    num_top_classes=8,
                )

        def on_preset_change(selected_preset_name: str):
            preset = TASK_PRESETS.get(selected_preset_name, {})
            resolved_img = download_if_url(preset.get("image"))
            return preset.get("labels", ""), resolved_img

        preset_selector.change( # type: ignore
            fn=on_preset_change,
            inputs=[preset_selector],
            outputs=[labels_input, input_img],
        )

        submit_btn.click( # type: ignore
            fn=run_zero_shot_classification,
            inputs=[input_img, labels_input],
            outputs=[output_label],
        )

    return app


if __name__ == "__main__":
    server = build_ui()
    server.launch(server_name="0.0.0.0", server_port=7861, share=True)