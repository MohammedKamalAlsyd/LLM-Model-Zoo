"""Gradio Web Server for PaliGemma 2 with interactive task examples.

Supports Captioning, VQA, Object Detection (<loc####>), and Segmentation (<seg###>).
"""

import os
import sys
import urllib.request
from pathlib import Path
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
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.model_loader import load_hf_model_weights

# Recommended: 10B mix checkpoint for accurate multi-task instruction following
HF_REPO = "google/paligemma2-10b-mix-448"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

# Path to cached demo images
EXAMPLES_DIR = Path(__file__).parent / "assets" / "examples"
EXAMPLE_IMAGES = {
    "animals.jpg": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats_and_dogs.jpg",
    "street.png": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
}


def ensure_example_assets() -> None:
    """Downloads lightweight sample images if not already present."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in EXAMPLE_IMAGES.items():
        filepath = EXAMPLES_DIR / filename
        if not filepath.exists():
            try:
                print(f"Downloading sample image: {filename}...")
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Notice: Could not download {filename}: {e}")


def get_model_and_pipeline():
    """Initializes model, preprocessor, and postprocessor."""
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
    postprocessor = PaliGemma2Postprocessor(tokenizer)
    return model, preprocessor, postprocessor


@torch.no_grad()
def generate(
    model: PaliGemma2ForConditionalGeneration,
    preprocessor: PaliGemma2Preprocessor,
    postprocessor: PaliGemma2Postprocessor,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 120,
    temp: float = 0.0,
) -> Tuple[str, Optional[Image.Image]]:
    """Runs generation and delegates coordinate parsing/rendering to postprocessor."""
    inputs = preprocessor(text=prompt, image=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)

    attention_mask = inputs["attention_mask"].to(DEVICE)
    kv_cache = KVCache()
    generated_ids = []

    # 1. Prefill Step
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
    )
    next_logits = outputs["logits"][:, -1, :]
    eos_ids = model.config.eos_token_ids

    # 2. Decode Loop
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


def main():
    ensure_example_assets()
    model, preprocessor, postprocessor = get_model_and_pipeline()

    def run_inference(image, prompt, max_new, temp):
        if not image:
            return "Please provide an image.", None
        prompt = prompt or "caption en"
        try:
            return generate(model, preprocessor, postprocessor, image, prompt, int(max_new), float(temp))
        except Exception as e:
            return f"Error: {e}", None

    with gr.Blocks(title="PaliGemma 2") as app:
        gr.Markdown(f"## PaliGemma 2 — `{HF_REPO}` ({DEVICE.upper()})")
        gr.Markdown(
            "Select an example below or upload your own image to test **Captioning**, **VQA**, "
            "**Object Detection**, and **Segmentation**."
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="caption en",
                    placeholder="e.g. 'caption en', 'answer en <question>', 'detect <label>', 'segment <label>'",
                )
                with gr.Row():
                    tokens_slider = gr.Slider(10, 500, 120, step=1, label="Max Tokens")
                    temp_slider = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="Temperature")
                btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(label="Generated Text")
                output_img = gr.Image(type="pil", label="Visual Annotations (Detection / Segmentation)")

        btn.click( # type: ignore
            run_inference,
            inputs=[input_img, prompt_input, tokens_slider, temp_slider],
            outputs=[output_text, output_img],
        )

        # Example configurations representing each major task
        animals_path = str(EXAMPLES_DIR / "animals.jpg")
        street_path = str(EXAMPLES_DIR / "street.png")

        candidate_examples = [
            # [Image, Prompt, Max Tokens, Temp]
            [animals_path, "caption en", 64, 0.0],
            [animals_path, "answer en What animals are sitting together?", 64, 0.0],
            [animals_path, "detect cat ; dog", 128, 0.0],
            [animals_path, "segment dog", 128, 0.0],
            [street_path, "describe en", 128, 0.0],
            [street_path, "detect person ; car", 128, 0.0],
        ]

        valid_examples = [ex for ex in candidate_examples if os.path.exists(ex[0])]
        if valid_examples:
            gr.Examples(
                examples=valid_examples,
                inputs=[input_img, prompt_input, tokens_slider, temp_slider],
                outputs=[output_text, output_img],
                fn=run_inference,
                cache_examples=False,
            )

    app.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()