"""Interactive Gradio Web Server for Ministral-3 Multimodal.

Supports both multimodal (Image + Text) and pure text-only conversations
with autoregressive KV-cached decoding.
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple
import torch
import gradio as gr
from transformers import AutoProcessor

# Setup pathing to allow absolute imports from 'Zoo'
current_path = Path(__file__).resolve()
root_node = next(p for p in current_path.parents if (p / "Zoo").exists())
if str(root_node) not in sys.path:
    sys.path.insert(0, str(root_node))

from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights
from Zoo.Ministral3.configs import Ministral3MultimodalConfig
from Zoo.Ministral3.Ministral3Multimodal import Mistral3ForConditionalGeneration

# --- Constants & Configuration ---
HF_REPO = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
CONFIG_FILE = Path(__file__).parent / "config.json"


def load_model_and_processor() -> Tuple[Mistral3ForConditionalGeneration, Any, str, torch.dtype]:
    """Loads model configuration, streams safetensors shards, and initializes processor."""
    device, dtype = auto_detect_device_and_dtype()
    print(f"Initializing Ministral-3 Multimodal on {device.upper()} in {dtype}...")

    # Load configuration
    if CONFIG_FILE.exists():
        config = Ministral3MultimodalConfig.from_json_file(CONFIG_FILE)
    else:
        # Fallback to defaults matching 3B checkpoint
        config = Ministral3MultimodalConfig()

    model = Mistral3ForConditionalGeneration(config)

    # Universal shard-by-shard streaming loader with global strict key verification
    load_hf_model_weights(
        model=model,
        repo_id=HF_REPO,
        strict=True,
        device=device,
        dtype=dtype,
    )

    print("Loading Hugging Face AutoProcessor...")
    processor = AutoProcessor.from_pretrained(HF_REPO)
    return model, processor, device, dtype


# Initialize globally
model, processor, DEVICE, DTYPE = load_model_and_processor()


@torch.no_grad()
def generate(
    image: Any,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    repetition_penalty: float = 1.15,
) -> str:
    """Runs generation supporting both Text-Only and Multimodal prompts."""
    if not prompt.strip() and image is None:
        return "Please enter a prompt or upload an image."

    # 1. System Prompt prevents the model from falling back to text-only RLHF disclaimers
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an advanced multimodal AI assistant. You can see, analyze, "
                "and describe images directly and with high precision."
            ),
        }
    ]

    if image is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        })

    formatted_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2. Tokenize and preprocess
    if image is not None:
        inputs = processor(images=image, text=formatted_text, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)
        image_sizes = inputs.get("image_sizes", None)
        if image_sizes is not None:
            image_sizes = image_sizes.to(DEVICE)
    else:
        inputs = processor(text=formatted_text, return_tensors="pt")
        pixel_values = None
        image_sizes = None

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # =========================================================================
    # 3. Prefill Phase
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
    eos_token_id = processor.tokenizer.eos_token_id

    # =========================================================================
    # 4. Decode Phase with Repetition Penalty
    # =========================================================================
    for _ in range(max_tokens):
        # Apply repetition penalty to prevent punctuation stuttering
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
        if token_id == eos_token_id:
            break

        generated_tokens.append(token_id)

        outputs = model(
            input_ids=next_token,
            pixel_values=None,
            image_sizes=None,
            past_key_values=kv_cache,
            logits_to_keep=1,
        )
        next_token_logits = outputs["logits"][:, -1, :]

    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# =============================================================================
# Gradio Interface
# =============================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Ministral-3 Zoo") as demo:
        gr.Markdown(
            f"""
            # 🦁 Ministral-3 Multimodal (3B)
            **Clean-Room PyTorch Implementation** running on **{DEVICE.upper()}** ({DTYPE}).
            - **Multimodal & Text-Only**: Upload an image to analyze it, or leave empty to chat directly.
            - **Vision Encoder**: Native 2D Axial RoPE Pixtral Vision Tower
            - **Projector**: Spatial 2x2 Patch Merger + MLP
            - **Language Backbone**: YaRN RoPE & LLaMA-4 Scaled GQA Decoder
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image (Optional)")
                prompt_input = gr.Textbox(
                    label="User Prompt",
                    placeholder="Ask a question or describe an image...",
                    value="Hello! What can you do?",
                    lines=3,
                )

                with gr.Accordion("Generation Parameters", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=16, maximum=1024, value=256, step=16, label="Max New Tokens"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.2, value=0.1, step=0.05, label="Temperature"
                    )

                submit_btn: Any = gr.Button("Generate Response", variant="primary")

                gr.Examples(
                    examples=[
                        ["Describe this image in detail."],
                        ["Transcribe any text visible in this image."],
                        ["Explain the difference between supervised and unsupervised learning."],
                    ],
                    inputs=[prompt_input],
                )

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Model Output", lines=12)

        submit_btn.click(
            fn=generate,
            inputs=[input_image, prompt_input, max_tokens_slider, temp_slider],
            outputs=[output_text],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)