"""Gradio Web Server for PaliGemma 2."""

import os
import sys
import torch
import gradio as gr
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Environment & Path Setup
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Allows 'from common...'
sys.path.append(os.path.dirname(__file__))

from Zoo.PaliGemma2.configs import PaliGemma2Config
from Zoo.PaliGemma2.PaliGemma2 import PaliGemma2ForConditionalGeneration
from Zoo.PaliGemma2.preprocessor.PaliGemma2Processor import PaliGemma2Processor
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.model_loader import load_hf_model_weights

HF_REPO = "google/paligemma2-3b-pt-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32


def get_model_and_processor():
    config = PaliGemma2Config()
    model = PaliGemma2ForConditionalGeneration(config)

    # Downloads to ~/.cache/huggingface/hub and loads weights directly
    cache_dir = load_hf_model_weights(
        model=model,
        repo_id=HF_REPO,
        strict=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    model.tie_weights()

    tokenizer = AutoTokenizer.from_pretrained(cache_dir, padding_side="right")
    processor = PaliGemma2Processor(tokenizer)
    return model, processor


@torch.no_grad()
def generate(model, processor, image, prompt, max_tokens=100, temp=0.7):
    inputs = processor(text=prompt, image=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)
    
    attention_mask = inputs["attention_mask"].to(DEVICE)
    kv_cache = KVCache()
    generated_ids = []

    # 1. Prefill Step (Image + Prompt tokens)
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
    )
    next_logits = outputs["logits"][:, -1, :]

    # 2. Decode Loop (Token by token)
    for _ in range(max_tokens):
        if temp > 0:
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        if next_token.item() == model.config.text_config.eos_token_id:
            break

        generated_ids.append(next_token.item())
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=DEVICE)], dim=1)

        outputs = model(
            input_ids=next_token,
            pixel_values=None,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        next_logits = outputs["logits"][:, -1, :]

    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    model, processor = get_model_and_processor()

    def run_inference(image, text, max_new, temp):
        if not image:
            return "Please upload an image."
        text = text or "describe this image"
        try:
            return generate(model, processor, image, text, int(max_new), float(temp))
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title="PaliGemma 2") as app:
        gr.Markdown(f"### PaliGemma 2 (3B) on {DEVICE.upper()}")
        with gr.Row():
            img = gr.Image(type="pil", label="Input Image")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="describe this image")
                tokens = gr.Slider(10, 500, 100, step=1, label="Max Tokens")
                temp = gr.Slider(0.0, 1.5, 0.7, step=0.1, label="Temperature")
                btn = gr.Button("Generate", variant="primary")
                out = gr.Textbox(label="Output")

        btn.click(run_inference, [img, prompt, tokens, temp], out) # type: ignore

    app.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()