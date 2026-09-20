"""Interactive Gradio Web Server for Ministral-3 Multimodal.

Supports on-the-fly preset loading, visual QA, landmark identification,
exhaustive scene description, OCR transcription, and pure text reasoning.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor
from dotenv import load_dotenv

load_dotenv()

# Setup pathing to allow absolute imports from repository root
current_path = Path(__file__).resolve()
root_node = next(p for p in current_path.parents if (p / "Zoo").exists())
if str(root_node) not in sys.path:
    sys.path.insert(0, str(root_node))

from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights
from Zoo.Ministral3.configs import Ministral3MultimodalConfig
from Zoo.Ministral3.Ministral3Multimodal import Mistral3ForConditionalGeneration

# --- Constants & Global Configuration ---
HF_REPO = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
CONFIG_FILE = Path(__file__).parent / "config.json"
DEVICE, DTYPE = auto_detect_device_and_dtype()

# Reliable public reference images (Wikimedia standard 960px bucket and COCO val2017)
PRESET_CONFIG = {
    "Landmark - Great Pyramid of Giza": {
        # Wikimedia requires strict thumbnail bucket sizes (960px is an allowed size)
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Kheops-Pyramid.jpg/960px-Kheops-Pyramid.jpg",
        "prompt": "Identify this landmark and describe its architectural and visual features in detail.",
        "max_tokens": 256,
        "temperature": 0.1,
    },
    "VQA - Two Sleeping Cats": {
        "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "prompt": "What animals are visible on the couch, and what electronic accessories are lying nearby?",
        "max_tokens": 128,
        "temperature": 0.1,
    },
    "Detailed Scene - Living Room": {
        "image": "http://images.cocodataset.org/val2017/000000000139.jpg",
        "prompt": "Provide a comprehensive, high-precision description of this room, its furniture, and ambient lighting.",
        "max_tokens": 256,
        "temperature": 0.1,
    },
    "OCR & Scene Understanding - Street Clock": {
        "image": "http://images.cocodataset.org/val2017/000000000285.jpg",
        "prompt": "Describe what is shown in this outdoor street scene, noting any prominent structures or time displays.",
        "max_tokens": 160,
        "temperature": 0.1,
    },
    "Visual Reasoning - Kitchen Appliances": {
        "image": "http://images.cocodataset.org/val2017/000000000632.jpg",
        "prompt": "Analyze the state of this kitchen. What appliances are visible, and does the countertop appear occupied?",
        "max_tokens": 160,
        "temperature": 0.1,
    },
    "Pure Text - Technical Architecture": {
        "image": None,
        "prompt": "Explain the architectural difference between Grouped-Query Attention (GQA) and Multi-Head Attention (MHA). Why does GQA save memory?",
        "max_tokens": 256,
        "temperature": 0.2,
    },
}


def download_if_url(image_path: Optional[str]) -> Optional[str]:
    """Downloads remote image URLs to a local temporary cache on the fly with failure resilience."""
    if not image_path:
        return None
    if isinstance(image_path, str) and image_path.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "ministral3_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_filename = os.path.join(cache_dir, os.path.basename(image_path.split("?")[0]))

        # Return cached image if already present
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
            return local_filename

        print(f"Downloading sample image on the fly: {image_path}...")
        try:
            # Policy-compliant User-Agent prevents 400/403 errors from Wikimedia/COCO
            headers = {
                "User-Agent": "LLMModelZoo/1.0 (https://github.com/ModelZoo; contact@modelzoo.org) Python-urllib"
            }
            req = urllib.request.Request(image_path, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response, open(local_filename, "wb") as out_file:
                out_file.write(response.read())
            return local_filename
        except Exception as e:
            print(f"Warning: Could not download preset image from {image_path}: {e}")
            if os.path.exists(local_filename):
                try:
                    os.remove(local_filename)
                except OSError:
                    pass
            return None
    return image_path


def load_model_and_processor() -> Tuple[Mistral3ForConditionalGeneration, Any]:
    """Initializes Ministral-3 model configuration, weights, and processor."""
    print(f"Initializing Ministral-3 Multimodal on {DEVICE.upper()} in {DTYPE}...")

    if CONFIG_FILE.exists():
        config = Ministral3MultimodalConfig.from_json_file(CONFIG_FILE)
    else:
        config = Ministral3MultimodalConfig()

    model = Mistral3ForConditionalGeneration(config)

    # Universal shard streaming loader matching checkpoints 1:1
    load_hf_model_weights(
        model=model,
        repo_id=HF_REPO,
        strict=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("Loading AutoProcessor...")
    processor = AutoProcessor.from_pretrained(HF_REPO)
    return model, processor


# Initialize model and processor globally
MODEL, PROCESSOR = load_model_and_processor()


@torch.no_grad()
def generate(
    model: Mistral3ForConditionalGeneration,
    processor: Any,
    image: Optional[Image.Image],
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.1,
    repetition_penalty: float = 1.0,  # 1.0 prevents synthetic word degradation
) -> str:
    """Runs conditioned autoregressive inference supporting text and image-text inputs."""
    if not prompt.strip() and image is None:
        return "Please provide an image or enter a text prompt."

    # 1. Structure message format aligned with HF multimodal chat templates
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception:
            formatted_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(images=image, text=formatted_text, return_tensors="pt")

        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)

        image_sizes = inputs.get("image_sizes", None)
        if image_sizes is not None and isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.to(DEVICE)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=formatted_text, return_tensors="pt")
        pixel_values = None
        image_sizes = None

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # =========================================================================
    # 2. Prefill Phase
    # =========================================================================
    kv_cache = KVCache()
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
        past_key_values=kv_cache,
        logits_to_keep=1,
    )

    next_token_logits = outputs["logits"][:, -1, :]
    generated_tokens = []

    # Identify stop tokens
    eos_token_id = processor.tokenizer.eos_token_id
    stop_token_ids = {eos_token_id}
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "eos_token_id"):
        stop_token_ids.add(model.config.text_config.eos_token_id)

    # =========================================================================
    # 3. Autoregressive Decode Phase
    # =========================================================================
    for _ in range(max_tokens):
        if repetition_penalty != 1.0 and generated_tokens:
            for prev_token in set(generated_tokens):
                if next_token_logits[0, prev_token] < 0:
                    next_token_logits[0, prev_token] *= repetition_penalty
                else:
                    next_token_logits[0, prev_token] /= repetition_penalty

        if temperature > 0.0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id in stop_token_ids:
            break

        generated_tokens.append(token_id)

        # Autoregressive forward step bypasses vision tower
        outputs = model(
            input_ids=next_token,
            pixel_values=None,
            image_sizes=None,
            past_key_values=kv_cache,
            logits_to_keep=1,
        )
        next_token_logits = outputs["logits"][:, -1, :]

    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def on_preset_change(preset_name: str) -> Tuple[str, Optional[str], int, float]:
    """Swaps prompt, image preview, and sampling parameters when a preset is selected."""
    cfg = PRESET_CONFIG.get(preset_name, {})
    resolved_img_path = download_if_url(cfg.get("image", None))
    return (
        cfg.get("prompt", ""),
        resolved_img_path,
        cfg.get("max_tokens", 256),
        cfg.get("temperature", 0.1),
    )


def main():
    default_preset = "Landmark - Great Pyramid of Giza"
    default_cfg = PRESET_CONFIG[default_preset]
    default_img_path = download_if_url(default_cfg["image"])

    def run_inference(image_input: Any, prompt_text: str, max_new_tokens: int, temp: float) -> str:
        if not image_input and not prompt_text.strip():
            return "Please enter a prompt or upload an image."

        if isinstance(image_input, str) and image_input.strip():
            resolved = download_if_url(image_input)
            img = Image.open(resolved).convert("RGB") if resolved else None
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = None

        prompt_text = prompt_text or ("Describe this image in detail." if img is not None else "Hello!")
        try:
            return generate(
                model=MODEL,
                processor=PROCESSOR,
                image=img,
                prompt=prompt_text,
                max_tokens=int(max_new_tokens),
                temperature=float(temp),
            )
        except Exception as e:
            return f"Error during generation: {e}"

    with gr.Blocks(title="Ministral-3 Multimodal") as app:
        gr.Markdown(
            f"""
            # 🦁 Ministral-3 Multimodal — `{HF_REPO}` ({DEVICE.upper()})
            **Clean-Room PyTorch Implementation** running in **{DTYPE}**.
            - **Vision Encoder**: Native 2D Axial RoPE Pixtral Vision Tower ($14 \\times 14$ patches).
            - **Projector**: Spatial $2 \\times 2$ Patch Merger + Projection MLP.
            - **Language Backbone**: Ministral-3 (3B) with YaRN RoPE & LLaMA-4 Attn Query Scaling.
            """
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_CONFIG.keys()),
                    value=default_preset,
                    label="Task Presets (Auto-loads sample image & prompt on the fly)",
                )

                input_img = gr.Image(
                    type="pil",
                    value=default_img_path,
                    label="Input Image (Upload or select from presets)",
                )

                prompt_input = gr.Textbox(
                    label="User Prompt",
                    value=default_cfg["prompt"],
                    placeholder="Enter a prompt or question about the image...",
                    lines=3,
                )

                with gr.Row():
                    tokens_slider = gr.Slider(
                        minimum=16,
                        maximum=1024,
                        value=default_cfg["max_tokens"],
                        step=16,
                        label="Max Tokens",
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=default_cfg["temperature"],
                        step=0.05,
                        label="Temperature",
                    )

                submit_btn = gr.Button("Generate Response", variant="primary")

            # Right Column: Outputs
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Model Output", lines=15)

        # On Preset Change: Dynamically update image, prompt, and sliders
        preset_dropdown.change( # type:ignore
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[prompt_input, input_img, tokens_slider, temp_slider],
            show_progress="hidden",
        )

        # Run Inference
        submit_btn.click( # type:ignore
            fn=run_inference,
            inputs=[input_img, prompt_input, tokens_slider, temp_slider],
            outputs=[output_text],
        )

    app.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()