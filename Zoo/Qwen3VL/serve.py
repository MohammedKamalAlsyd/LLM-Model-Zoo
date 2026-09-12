import gradio as gr
from gradio.themes import Soft
from gradio.themes.utils.fonts import GoogleFont
import torch
from transformers import AutoProcessor
from PIL import Image

from .utils.model_loader import load_qwen3_vl

# Configuration
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Initializing Qwen3-VL on {DEVICE}...")
model = load_qwen3_vl(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


def chat_response(message, history, temperature, top_p, max_new_tokens):
    text_prompt = message.get("text", "")
    files = message.get("files", [])

    messages = [{"role": "user", "content": []}]
    image_obj = None

    if files:
        file_path = files[0] if isinstance(files, list) else files
        image_obj = Image.open(file_path).convert("RGB")
        messages[0]["content"].append({"type": "image", "image": image_obj})

    if text_prompt.strip():
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
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    ):
        decoded = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        response_text += decoded
        yield response_text


# Predefined Examples formatted as [message_dict, temperature, top_p, max_new_tokens]
sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

examples = [
    [
        {"text": "Describe this image in detail and tell me what the cat is doing.", "files": [sample_image_url]},
        0.7,
        0.9,
        512,
    ],
    [
        {"text": "Extract and list all the prominent visual colors and objects in this scene.", "files": [sample_image_url]},
        0.7,
        0.9,
        512,
    ],
    [
        {"text": "Write a creative short story inspired by this image.", "files": [sample_image_url]},
        0.7,
        0.9,
        512,
    ],
    [
        {"text": "Explain how 3D Multimodal RoPE (M-RoPE) works in Qwen3-VL in simple terms.", "files": []},
        0.7,
        0.9,
        512,
    ],
]

custom_css = """
#main-header {
    text-align: center;
    margin-bottom: 1.5rem;
}
#badge-container {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.badge {
    background: linear-gradient(135deg, #1e293b, #334155);
    color: #38bdf8;
    padding: 0.35rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid #475569;
}
"""

theme = Soft(
    primary_hue="sky",
    secondary_hue="slate",
    font=[GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(theme=theme, css=custom_css, title="Qwen3-VL 4B Instruct") as demo:
    with gr.Column(elem_id="main-header"):
        gr.Markdown(
            """
            # 👁️ Qwen3-VL 4B Instruct
            ### Pure Standalone PyTorch Implementation with DeepStack & M-RoPE
            """
        )
        gr.HTML(
            """
            <div id="badge-container">
                <span class="badge">🚀 4B Parameters</span>
                <span class="badge">🧠 DeepStack Feature Fusion</span>
                <span class="badge">📐 3D Multimodal RoPE</span>
                <span class="badge">⚡ Zero HuggingFace Base Class Dependency</span>
            </div>
            """
        )

    with gr.Accordion("⚙️ Generation Settings", open=False):
        with gr.Row():
            temperature = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.7, step=0.05, label="Temperature", info="Controls randomness"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P", info="Nucleus sampling threshold"
            )
            max_new_tokens = gr.Slider(
                minimum=64, maximum=2048, value=512, step=64, label="Max Output Tokens", info="Maximum tokens to generate"
            )

    chat_interface = gr.ChatInterface(
        fn=chat_response,
        type="messages",  # type: ignore
        multimodal=True,
        additional_inputs=[temperature, top_p, max_new_tokens],
        textbox=gr.MultimodalTextbox(
            file_types=["image"],
            placeholder="Type your message or drag & drop an image here...",
            scale=8,
        ),
        examples=examples,
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)