import torch
import numpy as np
from PIL import Image
import gradio as gr
from typing import Optional

from Zoo.SAM3.SAM3 import Sam3Model
from Zoo.SAM3.SubModels.SAM3Processor import SAM3Processor
from Zoo.SAM3.utils.model_loader import load_sam3_from_hf

device = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_MODEL: Optional[Sam3Model] = None
GLOBAL_PROCESSOR: Optional[SAM3Processor] = None

EXAMPLES = [
    [
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800",
        "dog",
        0.35,
    ],
    [
        "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=800",
        "sports car",
        0.30,
    ],
    [
        "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=800",
        "yellow bandana",
        0.25,
    ],
    [
        "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800",
        "vintage car wheel",
        0.30,
    ],
]


def initialize_app():
    global GLOBAL_MODEL, GLOBAL_PROCESSOR
    if GLOBAL_MODEL is None:
        print(f"Loading SAM3 Model & Processor on {device.upper()}...")
        GLOBAL_MODEL = load_sam3_from_hf(device=device)
        GLOBAL_PROCESSOR = SAM3Processor()
        print("Ready!")


def draw_overlay(image: Image.Image, masks: torch.Tensor) -> np.ndarray:
    """Overlays predicted masks onto the input image with high contrast colors."""
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    colors = [
        [255, 60, 60],
        [60, 255, 60],
        [60, 120, 255],
        [255, 215, 0],
        [255, 60, 255],
        [0, 255, 255],
        [255, 140, 0],
    ]

    for i in range(min(len(masks), 25)):
        mask = masks[i].numpy()
        if not mask.any():
            continue
        color = np.array(colors[i % len(colors)], dtype=np.float32)
        img_np[mask] = img_np[mask] * 0.45 + color * 0.55

    return np.clip(img_np, 0, 255).astype(np.uint8)


def predict(image: Image.Image, text_prompt: str, score_threshold: float):
    if image is None:
        return None, "Please upload or select an image."
    if not text_prompt.strip():
        text_prompt = "object"

    if GLOBAL_MODEL is None or GLOBAL_PROCESSOR is None:
        raise RuntimeError("Model or Processor not initialized.")

    pixel_values, orig_size = GLOBAL_PROCESSOR.process_image(image, device=device)
    input_ids, attention_mask = GLOBAL_PROCESSOR.process_text(text_prompt, device=device)

    with torch.no_grad():
        outputs = GLOBAL_MODEL(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    raw_masks = outputs["pred_masks"][0]
    scores = outputs["pred_logits"][0].sigmoid()
    if outputs["presence_logits"] is not None:
        scores = scores * outputs["presence_logits"][0].sigmoid()

    keep = scores > score_threshold
    filtered_masks = raw_masks[keep]

    if filtered_masks.shape[0] == 0:
        return image, f"No instances found matching '{text_prompt}' with score > {score_threshold:.2f}."

    binary_masks = GLOBAL_PROCESSOR.post_process_masks(
        filtered_masks.unsqueeze(0),
        orig_size=orig_size,
        threshold=0.0,
    )[0]

    annotated_image = draw_overlay(image, binary_masks)
    return annotated_image, f"Detected {filtered_masks.shape[0]} instance(s) matching '{text_prompt}'."


def launch_ui():
    initialize_app()

    with gr.Blocks(title="SAM3 Open-Vocabulary Segmenter") as demo:
        gr.Markdown(
            f"# 🎯 SAM3 Open-Vocabulary Segmenter (From Scratch)\n"
            f"Running natively on **{device.upper()}** without wrapper conversion layers."
        )
        gr.Markdown(
            "Segment any concept from text prompts, automatically localize objects, "
            "or test predefined examples below."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                text_input = gr.Textbox(
                    label="Text Prompt",
                    value="dog",
                    placeholder="e.g. dog, sports car, wheel, bandana...",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.35,
                    step=0.05,
                    label="Confidence Threshold",
                )
                submit_btn = gr.Button("Segment Objects", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Segmented Output")
                output_text = gr.Textbox(label="Status / Detection Summary")

        gr.Markdown("### Examples")
        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_image, text_input, threshold_slider],
            outputs=[output_image, output_text],
            fn=predict,
            cache_examples=False,
        )

        submit_btn.click(
            fn=predict,
            inputs=[input_image, text_input, threshold_slider],
            outputs=[output_image, output_text],
        )

    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    launch_ui()