import gradio as gr
import torch
from transformers import AutoProcessor

from .utils.model_loader import load_qwen3_vl

# Configuration
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Qwen3-VL on {DEVICE}...")
model = load_qwen3_vl(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)


def chat(message, history):
    text_prompt = message["text"]
    images = message.get("files", [])

    messages = [{"role": "user", "content": []}]
    image_obj = None

    if images:
        from PIL import Image
        image_obj = Image.open(images[0]).convert("RGB")
        messages[0]["content"].append({"type": "image", "image": image_obj})

    messages[0]["content"].append({"type": "text", "text": text_prompt})

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[image_obj] if image_obj else None,
        return_tensors="pt",
    ).to(DEVICE)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    mm_token_type_ids = inputs.get("mm_token_type_ids", None)

    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=torch.bfloat16)

    response_text = ""
    for token_id in model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        max_new_tokens=512,
        temperature=0.7,
    ):
        decoded = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        response_text += decoded
        yield response_text


with gr.Blocks(title="Qwen3-VL Instruct") as demo:
    gr.Markdown("## Qwen3-VL-4B-Instruct (Pure Standalone PyTorch)")
    gr.ChatInterface(
        fn=chat,
        multimodal=True,
        textbox=gr.MultimodalTextbox(file_types=["image"], placeholder="Ask a question or upload an image..."),
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)