import torch
from PIL import Image
import gradio as gr
from typing import Optional

from Zoo.CLIP.utils.model_loader import load_clip_from_hf
from Zoo.CLIP.SubModels.CLIPProcessor import CLIPProcessor
from Zoo.CLIP.CLIP import CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_MODEL: Optional[CLIPModel] = None
GLOBAL_PROCESSOR: Optional[CLIPProcessor] = None

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


def initialize_app():
    global GLOBAL_MODEL, GLOBAL_PROCESSOR

    print("Loading Model and Processor...")
    GLOBAL_MODEL = load_clip_from_hf(device=device)
    GLOBAL_PROCESSOR = CLIPProcessor()
    print("Ready!")


def predict(image: Image.Image, classes_text: str):
    if GLOBAL_MODEL is None or GLOBAL_PROCESSOR is None:
        raise RuntimeError("Model or processor was not properly initialized.")

    if image is None:
        return {"Error: Please upload an image": 1.0}

    if not classes_text.strip():
        return {"Error: Please enter at least one class": 1.0}

    image = image.convert("RGB")
    classes = [c.strip() for c in classes_text.split(",") if c.strip()]

    pixel_values = GLOBAL_PROCESSOR.process_image(image, device)
    input_ids = GLOBAL_PROCESSOR.process_text(classes, device)

    with torch.no_grad():
        logits_per_image, _ = GLOBAL_MODEL(input_ids, pixel_values)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    return dict(
        sorted(
            {classes[i]: float(probs[i]) for i in range(len(classes))}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )


def load_example(index):
    return EXAMPLES[index]


def launch_ui():
    initialize_app()

    with gr.Blocks(title="CLIP Zero-Shot Classifier") as demo:
        gr.Markdown(
            f"# 🔍 CLIP Zero-Shot Image Classification (From Scratch)\n"
            f"Running natively on **{device.upper()}**."
        )
        gr.Markdown(
            "Upload any image and type any categories you can think of. "
            "CLIP calculates the similarity between the image and the text!"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")

                input_classes = gr.Textbox(
                    label="Categories (Comma-Separated)",
                    placeholder="e.g., a photo of a dog, a photo of a cat...",
                    value=EXAMPLES[0][1],
                )

                submit_btn = gr.Button("Classify Image", variant="primary")

            with gr.Column():
                output_label = gr.Label(label="AI Predictions (Probabilities)")

        gr.Markdown("### Examples")

        with gr.Row():
            for i, (_, classes) in enumerate(EXAMPLES):
                gr.Button(f"Example {i + 1}").click(
                    fn=lambda i=i: load_example(i),
                    outputs=[input_image, input_classes],
                )

        submit_btn.click(
            fn=predict,
            inputs=[input_image, input_classes],
            outputs=output_label,
        )

    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    launch_ui()