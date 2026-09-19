"""Gradio Web Server for PaliGemma 2 with on-the-fly preset loading.

Supports Captioning, VQA, Object Detection (<loc####>), and Segmentation (<seg###>).
"""

import os
import sys
import tempfile
import urllib.request
from typing import Optional, Tuple
import gradio as gr
from PIL import Image
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from Zoo.PaliGemma2.configs import PaliGemma2Config
from Zoo.PaliGemma2.PaliGemma2 import PaliGemma2ForConditionalGeneration
from Zoo.PaliGemma2.processing.PaliGemma2Preprocessor import PaliGemma2Preprocessor
from Zoo.PaliGemma2.processing.PaliGemma2Postprocessor import PaliGemma2Postprocessor
from Zoo.PaliGemma2.modules.MaskDecoder import load_mask_decoder
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.model_loader import load_hf_model_weights

HF_REPO = "google/paligemma2-3b-mix-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

# On-the-fly task configuration with reliable public COCO reference samples
PRESET_CONFIG = {
    "VQA - Two Cats": {
        "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "prompt": "answer en What animals are lying on the couch?",
        "max_tokens": 64,
        "temperature": 0.0,
    },
    "Caption - Two Cats": {
        "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "prompt": "caption en",
        "max_tokens": 64,
        "temperature": 0.0,
    },
    "Object Detection - Cats": {
        "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "prompt": "detect cat ; couch ; remote",
        "max_tokens": 128,
        "temperature": 0.0,
    },
    "Segmentation - Cats": {
        "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "prompt": "segment cat on the left",
        "max_tokens": 128,
        "temperature": 0.0,
    },
    "Object Detection - Living Room": {
        "image": "http://images.cocodataset.org/val2017/000000000139.jpg",
        "prompt": "detect chair ; dining table ; bottle",
        "max_tokens": 128,
        "temperature": 0.0,
    },
    "Detailed Description - Living Room": {
        "image": "http://images.cocodataset.org/val2017/000000000139.jpg",
        "prompt": "describe en",
        "max_tokens": 128,
        "temperature": 0.0,
    },
}


def download_if_url(image_path: Optional[str]) -> Optional[str]:
    """Downloads remote image URLs to a local temporary cache on the fly."""
    if not image_path:
        return None
    if isinstance(image_path, str) and image_path.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "paligemma2_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_filename = os.path.join(cache_dir, os.path.basename(image_path.split("?")[0]))
        if not os.path.exists(local_filename):
            print(f"Downloading sample image on the fly: {image_path}...")
            # Custom User-Agent header prevents 403 Forbidden errors
            req = urllib.request.Request(image_path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response, open(local_filename, "wb") as out_file:
                out_file.write(response.read())
        return local_filename
    return image_path


def get_model_and_pipeline():
    """Initializes PaliGemma 2 model and processor with downloaded weights."""
    config = PaliGemma2Config()
    model = PaliGemma2ForConditionalGeneration(config)

    cache_dir = load_hf_model_weights(
        model=model,
        repo_id=HF_REPO,
        strict=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    # Tie LM head to word embeddings (resolves missing lm_head.weight key)
    model.tie_weights()

    tokenizer = AutoTokenizer.from_pretrained(cache_dir, padding_side="right")
    preprocessor = PaliGemma2Preprocessor(tokenizer)
    
    # Load UViM mask decoder for instance segmentation
    mask_decoder = load_mask_decoder(device=DEVICE)
    postprocessor = PaliGemma2Postprocessor(tokenizer, mask_decoder=mask_decoder)
    return model, preprocessor, postprocessor


@torch.no_grad()
def generate(
    model: PaliGemma2ForConditionalGeneration,
    preprocessor: PaliGemma2Preprocessor,
    postprocessor: PaliGemma2Postprocessor,
    image: Optional[Image.Image],
    prompt: str,
    max_tokens: int = 120,
    temp: float = 0.0,
) -> Tuple[str, Optional[Image.Image]]:
    """Performs multimodal autoregressive generation and post-processes outputs."""
    inputs = preprocessor(text=prompt, image=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)

    attention_mask = inputs["attention_mask"].to(DEVICE)
    kv_cache = KVCache()
    generated_ids = []

    # 1. Prefill step
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
    )
    next_logits = outputs["logits"][:, -1, :]
    
    # Stop on EOS (1), End-of-turn (107), or Newline tokens (\n: 108 and 13)
    eos_ids = set(model.config.eos_token_ids) | {108, 13}

    # 2. Decode loop
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

    # 3. Postprocess text and visual annotations
    clean_text, annotated_image = postprocessor.process(generated_ids, image=image)
    return clean_text, annotated_image


def on_preset_change(preset_name: str):
    """Automatically swaps image on the fly, prompt, and sliders when preset changes."""
    cfg = PRESET_CONFIG.get(preset_name, {})
    resolved_img_path = download_if_url(cfg.get("image", ""))
    return (
        cfg.get("prompt", ""),
        resolved_img_path,
        cfg.get("max_tokens", 120),
        cfg.get("temperature", 0.0),
    )


def main():
    model, preprocessor, postprocessor = get_model_and_pipeline()

    default_preset = "VQA - Two Cats"
    default_cfg = PRESET_CONFIG[default_preset]
    default_img_path = download_if_url(default_cfg["image"])

    def run_inference(image, prompt, max_new, temp):
        if not image and not prompt.strip():
            return "Please provide an image or enter a text prompt.", None

        # Convert image string URL or local path to PIL Image
        if isinstance(image, str) and image.strip():
            image_path = download_if_url(image)
            image = Image.open(image_path) if image_path else None
        elif not isinstance(image, Image.Image):
            image = None

        prompt = prompt or ("caption en" if image is not None else "Hello!")
        try:
            return generate(model, preprocessor, postprocessor, image, prompt, int(max_new), float(temp))
        except Exception as e:
            return f"Error: {e}", None

    with gr.Blocks(title="PaliGemma 2") as app:
        gr.Markdown(f"## PaliGemma 2 — `{HF_REPO}` ({DEVICE.upper()})")
        gr.Markdown(
            "Select a preset from the dropdown to load an image and task on the fly, "
            "or upload your own custom image."
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column():
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_CONFIG.keys()),
                    value=default_preset,
                    label="Task Presets (Auto-loads image & prompt on the fly)"
                )

                input_img = gr.Image(
                    type="pil",
                    value=default_img_path,
                    label="Input Image (Upload or select preset)"
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=default_cfg["prompt"],
                    placeholder="e.g. 'caption en', 'answer en <question>', 'detect <label>', 'segment <label>'",
                )

                with gr.Row():
                    tokens_slider = gr.Slider(
                        10, 500,
                        value=default_cfg["max_tokens"],
                        step=1,
                        label="Max Tokens"
                    )
                    temp_slider = gr.Slider(
                        0.0, 1.0,
                        value=default_cfg["temperature"],
                        step=0.1,
                        label="Temperature"
                    )

                btn = gr.Button("Generate", variant="primary")

            # Right Column: Outputs
            with gr.Column():
                output_text = gr.Textbox(label="Generated Text", lines=4)
                output_img = gr.Image(type="pil", label="Visual Annotations (Detection / Segmentation)")

        # On Preset Change: Load sample image and prompt on the fly
        preset_dropdown.change( # type:ignore
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[prompt_input, input_img, tokens_slider, temp_slider],
            show_progress="hidden",
        )

        # Inference Trigger
        btn.click( # type:ignore
            run_inference,
            inputs=[input_img, prompt_input, tokens_slider, temp_slider],
            outputs=[output_text, output_img],
        )

    app.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()