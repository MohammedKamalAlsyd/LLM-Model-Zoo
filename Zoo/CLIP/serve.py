"""Interactive Gradio Web Studio for CLIP Zero-Shot Classification."""

from typing import Optional
import gradio as gr
from PIL import Image
import torch
import numpy as np

from Zoo.CLIP.CLIP import CLIPModel
from Zoo.CLIP.processing.CLIPProcessor import CLIPProcessor
from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights

GLOBAL_MODEL: Optional[CLIPModel] = None
GLOBAL_PROCESSOR: Optional[CLIPProcessor] = None
RESOLVED_DEVICE: str = "cpu"

REPO_ID = "openai/clip-vit-base-patch32"

EXAMPLES = [
    [
        "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400",
        "a photo of a dog, a photo of a cat, a picture of outer space, a fast car",
    ],
    [
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400",
        "a beautiful landscape, the earth from space, a dark room, microscopic cells",
    ],
    [
        "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400",
        "a vintage car, a modern sports car, a bicycle, a motorcycle",
    ],
]


def initialize_app() -> None:
    """Initializes models using the common streaming loader."""
    global GLOBAL_MODEL, GLOBAL_PROCESSOR, RESOLVED_DEVICE

    RESOLVED_DEVICE, target_dtype = auto_detect_device_and_dtype()
    print(f"Initializing CLIP on {RESOLVED_DEVICE.upper()} in {target_dtype}...")

    GLOBAL_MODEL = CLIPModel()
    load_hf_model_weights(
        model=GLOBAL_MODEL,
        repo_id=REPO_ID,
        allow_patterns=["*.safetensors", "*.bin"],
        strict=True,
        device=RESOLVED_DEVICE,
        dtype=target_dtype,
    )
    GLOBAL_PROCESSOR = CLIPProcessor(tokenizer_id=REPO_ID)
    print("✓ Model and processor initialized successfully.")


def predict(image: Optional[Image.Image], classes_text: str):
    """Inference endpoint executing zero-shot classification."""
    if GLOBAL_MODEL is None or GLOBAL_PROCESSOR is None:
        raise RuntimeError("Model or processor was not properly initialized.")

    if image is None:
        return {"Error: Please upload an image": 1.0}

    classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    if not classes:
        return {"Error: Please enter at least one class": 1.0}

    batch = GLOBAL_PROCESSOR(text=classes, images=image)
    pixel_values = batch["pixel_values"].to(RESOLVED_DEVICE, dtype=next(GLOBAL_MODEL.parameters()).dtype)
    input_ids = batch["input_ids"].to(RESOLVED_DEVICE)

    with torch.no_grad():
        outputs = GLOBAL_MODEL(input_ids=input_ids, pixel_values=pixel_values)
        probs = outputs["logits_per_image"].softmax(dim=-1).squeeze(0).float().cpu().numpy()

    return {classes[i]: float(probs[i]) for i in np.argsort(-probs)}


def launch_ui() -> None:
    initialize_app()

    with gr.Blocks(title="CLIP Zero-Shot Studio") as demo:
        gr.Markdown(
            f"# 🔍 CLIP Zero-Shot Image Classification\n"
            f"Dual-Tower Metric Architecture running on **{RESOLVED_DEVICE.upper()}**."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                input_classes = gr.Textbox(
                    label="Candidate Categories (Comma-Separated)",
                    value=EXAMPLES[0][1],
                )
                submit_btn = gr.Button("Classify Image", variant="primary")

            with gr.Column():
                output_label = gr.Label(label="Classification Probabilities")

        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_image, input_classes],
            outputs=output_label,
            fn=predict,
            cache_examples=False,
        )

        submit_btn.click(
            fn=predict,
            inputs=[input_image, input_classes],
            outputs=output_label,
        )

    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_ui()