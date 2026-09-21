"""Unified Multimodal Gradio Web Server for PaliGemma 2 and Ministral-3.

Features dynamic VRAM lazy loading, model swapping, task-based routing,
and automated bounding box/segmentation mask rendering.
"""

import gc
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer

# Setup repository path
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights

# --- Supported Models & HF Repositories ---
PALIGEMMA_REPO = "google/paligemma2-3b-mix-224"
MINISTRAL_REPO = "mistralai/Ministral-3-3B-Instruct-2512-BF16"

DEVICE, DTYPE = auto_detect_device_and_dtype()

# --- Task Presets by Model ---
TASK_PRESETS = {
    "PaliGemma 2 (3B)": {
        "Instance Segmentation": {
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "prompt": "segment cat on the left",
            "max_tokens": 128,
            "temperature": 0.0,
            "supports_segmentation": True,
        },
        "Object Detection (<loc####>)": {
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "prompt": "detect cat ; couch ; remote",
            "max_tokens": 128,
            "temperature": 0.0,
            "supports_segmentation": False,
        },
        "Visual Question Answering": {
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "prompt": "answer en What animals are lying on the couch?",
            "max_tokens": 64,
            "temperature": 0.0,
            "supports_segmentation": False,
        },
        "Image Captioning": {
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "prompt": "caption en",
            "max_tokens": 64,
            "temperature": 0.0,
            "supports_segmentation": False,
        },
    },
    "Ministral-3 (3B)": {
        "Visual Question Answering": {
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "prompt": "What animals are visible on the couch, and what electronic accessories are lying nearby?",
            "max_tokens": 128,
            "temperature": 0.1,
            "supports_segmentation": False,
        },
        "Landmark & Entity ID": {
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Kheops-Pyramid.jpg/960px-Kheops-Pyramid.jpg",
            "prompt": "Identify this landmark and describe its architectural and visual features in detail.",
            "max_tokens": 256,
            "temperature": 0.1,
            "supports_segmentation": False,
        },
        "Dense Scene & OCR Reading": {
            "image": "http://images.cocodataset.org/val2017/000000000285.jpg",
            "prompt": "Describe what is shown in this outdoor street scene, noting any prominent structures or time displays.",
            "max_tokens": 160,
            "temperature": 0.1,
            "supports_segmentation": False,
        },
        "Pure Text Reasoning (No Image)": {
            "image": None,
            "prompt": "Explain the architectural difference between Grouped-Query Attention (GQA) and Multi-Head Attention (MHA).",
            "max_tokens": 256,
            "temperature": 0.2,
            "supports_segmentation": False,
        },
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
            req = urllib.request.Request(image_path, headers={"User-Agent": "LLMModelZoo/1.0 Python-urllib"})
            with urllib.request.urlopen(req, timeout=15) as response, open(local_filename, "wb") as out_file:
                out_file.write(response.read())
            return local_filename
        except Exception as e:
            print(f"Warning: Could not download image {image_path}: {e}")
            return None
    return image_path


class ModelManager:
    """Manages VRAM allocation and dynamic swapping between multimodal backbones."""

    def __init__(self):
        self.active_model_name: Optional[str] = None
        self.model: Optional[torch.nn.Module] = None
        self.processor: Any = None
        self.postprocessor: Any = None

    def unload(self):
        """Purges the active model from VRAM and triggers garbage collection."""
        if self.model is not None:
            print(f"Unloading '{self.active_model_name}' to free VRAM...")
            del self.model
            del self.processor
            del self.postprocessor
            self.model = None
            self.processor = None
            self.postprocessor = None
            self.active_model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_model(self, model_name: str) -> Tuple[torch.nn.Module, Any, Any]:
        """Loads the requested model on demand if not already in memory."""
        if self.active_model_name == model_name and self.model is not None:
            return self.model, self.processor, self.postprocessor

        self.unload()

        print(f"Lazy loading '{model_name}' on {DEVICE.upper()} in {DTYPE}...")

        if model_name == "PaliGemma 2 (3B)":
            from Zoo.PaliGemma2.configs import PaliGemma2Config
            from Zoo.PaliGemma2.PaliGemma2 import PaliGemma2ForConditionalGeneration
            from Zoo.PaliGemma2.processing.PaliGemma2Preprocessor import PaliGemma2Preprocessor
            from Zoo.PaliGemma2.processing.PaliGemma2Postprocessor import PaliGemma2Postprocessor
            from Zoo.PaliGemma2.modules.MaskDecoder import load_mask_decoder

            cfg = PaliGemma2Config()
            model = PaliGemma2ForConditionalGeneration(cfg)
            cache_dir = load_hf_model_weights(model, repo_id=PALIGEMMA_REPO, strict=False, device=DEVICE, dtype=DTYPE)
            model.tie_weights()

            tokenizer = AutoTokenizer.from_pretrained(cache_dir, padding_side="right")
            processor = PaliGemma2Preprocessor(tokenizer)
            mask_decoder = load_mask_decoder(device=DEVICE)
            postprocessor = PaliGemma2Postprocessor(tokenizer, mask_decoder=mask_decoder)

        elif model_name == "Ministral-3 (3B)":
            from Zoo.Ministral3.configs import Ministral3MultimodalConfig
            from Zoo.Ministral3.Ministral3Multimodal import Mistral3ForConditionalGeneration

            cfg = Ministral3MultimodalConfig()
            model = Mistral3ForConditionalGeneration(cfg)
            load_hf_model_weights(model, repo_id=MINISTRAL_REPO, strict=True, device=DEVICE, dtype=DTYPE)
            processor = AutoProcessor.from_pretrained(MINISTRAL_REPO)
            postprocessor = None
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.active_model_name = model_name
        self.model = model
        self.processor = processor
        self.postprocessor = postprocessor
        return self.model, self.processor, self.postprocessor


MANAGER = ModelManager()


@torch.no_grad()
def infer_paligemma(model, preprocessor, postprocessor, image, prompt, max_tokens, temp):
    """Execution pathway for PaliGemma 2 (prefix injection + mask parsing)."""
    inputs = preprocessor(text=prompt, image=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)

    attention_mask = inputs["attention_mask"].to(DEVICE)
    kv_cache = KVCache()
    generated_ids = []

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
    )
    next_logits = outputs["logits"][:, -1, :]
    eos_ids = set(model.config.eos_token_ids) | {108, 13}

    for _ in range(max_tokens):
        if temp > 0.0:
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id in eos_ids:
            break

        generated_ids.append(token_id)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=DEVICE)], dim=1)

        outputs = model(
            input_ids=next_token,
            pixel_values=None,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        next_logits = outputs["logits"][:, -1, :]

    return postprocessor.process(generated_ids, image=image)


@torch.no_grad()
def infer_ministral(model, processor, image, prompt, max_tokens, temp):
    """Execution pathway for Ministral-3 (chat templating + 2x2 patch merging)."""
    if image is not None:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=chat_text, images=image, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)
        image_sizes = inputs.get("image_sizes")
        if image_sizes is not None and isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.to(DEVICE)
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=chat_text, return_tensors="pt")
        pixel_values, image_sizes = None, None

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    kv_cache = KVCache()
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
        past_key_values=kv_cache,
        logits_to_keep=1,
    )

    next_logits = outputs["logits"][:, -1, :]
    generated_tokens = []
    stop_ids = {processor.tokenizer.eos_token_id, model.config.text_config.eos_token_id}

    for _ in range(max_tokens):
        if temp > 0.0:
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id in stop_ids:
            break

        generated_tokens.append(token_id)
        outputs = model(
            input_ids=next_token,
            pixel_values=None,
            image_sizes=None,
            past_key_values=kv_cache,
            logits_to_keep=1,
        )
        next_logits = outputs["logits"][:, -1, :]

    clean_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return clean_text, None  # Ministral-3 does not produce segmentation masks


def run_inference(model_name: str, image_input: Any, prompt_text: str, max_tokens: int, temp: float):
    """Central router dispatching inputs to the active model."""
    if not prompt_text.strip() and not image_input:
        return "Please upload an image or provide a prompt.", None

    img: Optional[Image.Image] = None

    # Handle string path or URL safely with explicit null guard
    if isinstance(image_input, str) and image_input.strip():
        local_path = download_if_url(image_input)
        if local_path is not None and os.path.exists(local_path):
            img = Image.open(local_path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")

    model, processor, postprocessor = MANAGER.get_model(model_name)

    if model_name == "PaliGemma 2 (3B)":
        return infer_paligemma(model, processor, postprocessor, img, prompt_text, int(max_tokens), float(temp))
    else:
        return infer_ministral(model, processor, img, prompt_text, int(max_tokens), float(temp))


def build_ui():
    initial_model = "PaliGemma 2 (3B)"
    initial_task = "Instance Segmentation"
    initial_preset = TASK_PRESETS[initial_model][initial_task]
    initial_img_path = download_if_url(initial_preset["image"])

    with gr.Blocks(title="Unified Multimodal Zoo") as app:
        gr.Markdown("# 🦁 Universal Multimodal VLM Studio")
        gr.Markdown(
            f"Run **PaliGemma 2** or **Ministral-3** with **dynamic VRAM swapping** and automated detection/segmentation parsing."
        )

        with gr.Row():
            # Left Controls
            with gr.Column(scale=1):
                model_selector = gr.Radio(
                    choices=list(TASK_PRESETS.keys()),
                    value=initial_model,
                    label="Active Model (Lazy Loaded into VRAM on Demand)",
                )

                task_selector = gr.Dropdown(
                    choices=list(TASK_PRESETS[initial_model].keys()),
                    value=initial_task,
                    label="Task Preset",
                )

                input_img = gr.Image(
                    type="pil",
                    value=initial_img_path,
                    label="Input Image",
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=initial_preset["prompt"],
                    lines=3,
                )

                with gr.Row():
                    tokens_slider = gr.Slider(16, 512, value=initial_preset["max_tokens"], step=16, label="Max Tokens")
                    temp_slider = gr.Slider(0.0, 1.0, value=initial_preset["temperature"], step=0.05, label="Temperature")

                submit_btn = gr.Button("Generate", variant="primary")

            # Right Outputs
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Generated Text / Logits", lines=8)
                output_img = gr.Image(
                    type="pil",
                    label="Visual Annotations (Bounding Boxes & Segmentation Masks)",
                    visible=True,
                )

        # Dynamic UI event handlers
        def on_model_change(selected_model):
            available_tasks = list(TASK_PRESETS[selected_model].keys())
            first_task = available_tasks[0]
            preset = TASK_PRESETS[selected_model][first_task]
            img_path = download_if_url(preset["image"])
            seg_visible = selected_model == "PaliGemma 2 (3B)"
            return (
                gr.update(choices=available_tasks, value=first_task),
                preset["prompt"],
                img_path,
                preset["max_tokens"],
                preset["temperature"],
                gr.update(visible=seg_visible),
            )

        def on_task_change(selected_model, selected_task):
            preset = TASK_PRESETS[selected_model].get(selected_task, {})
            img_path = download_if_url(preset.get("image", None))
            return (
                preset.get("prompt", ""),
                img_path,
                preset.get("max_tokens", 128),
                preset.get("temperature", 0.0),
            )

        model_selector.change( # type:ignore
            fn=on_model_change,
            inputs=[model_selector],
            outputs=[task_selector, prompt_input, input_img, tokens_slider, temp_slider, output_img],
        )

        task_selector.change( # type:ignore
            fn=on_task_change,
            inputs=[model_selector, task_selector],
            outputs=[prompt_input, input_img, tokens_slider, temp_slider],
        )

        submit_btn.click( # type:ignore
            fn=run_inference,
            inputs=[model_selector, input_img, prompt_input, tokens_slider, temp_slider],
            outputs=[output_text, output_img],
        )

    return app


if __name__ == "__main__":
    server = build_ui()
    server.launch(server_name="0.0.0.0", server_port=7860, share=True)