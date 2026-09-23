"""Interactive Gradio Web Studio for Stable Diffusion v1.5 image synthesis."""

from typing import Optional
import gradio as gr
from PIL import Image
import torch

from Zoo.StableDiffusion.StableDiffusion import StableDiffusionModel
from Zoo.StableDiffusion.pipeline.StableDiffusionPipeline import StableDiffusionPipeline
from Zoo.Common.model_loader import auto_detect_device_and_dtype

GLOBAL_PIPELINE: Optional[StableDiffusionPipeline] = None
RESOLVED_DEVICE: str = "cpu"


def initialize_app() -> None:
    """Initializes models using common loader logic and precision detection."""
    global GLOBAL_PIPELINE, RESOLVED_DEVICE

    RESOLVED_DEVICE, target_dtype = auto_detect_device_and_dtype()
    print(f"Initializing Stable Diffusion on {RESOLVED_DEVICE.upper()} in {target_dtype}...")

    model = StableDiffusionModel.from_pretrained_weights(
        repo_id="runwayml/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.ckpt",
        device=RESOLVED_DEVICE,
        dtype=target_dtype,
    )
    GLOBAL_PIPELINE = StableDiffusionPipeline(model=model)
    print("✓ Stable Diffusion pipeline initialized.")


def predict(
    prompt: str,
    uncond_prompt: str,
    input_image: Optional[Image.Image],
    strength: float,
    cfg_scale: float,
    steps: float,
    seed: float,
) -> Image.Image:
    """Inference endpoint connecting UI controls to the generation engine."""
    if GLOBAL_PIPELINE is None:
        raise RuntimeError("Generation pipeline is uninitialized.")

    active_seed = None if int(seed) == -1 else int(seed)

    return GLOBAL_PIPELINE(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        num_inference_steps=int(steps),
        seed=active_seed,
    )


def launch_ui() -> None:
    initialize_app()

    with gr.Blocks(title="Stable Diffusion v1.5 Studio") as demo:
        gr.Markdown(
            f"# 🎨 Stable Diffusion v1.5 (Clean-Room PyTorch)\n"
            f"Execution running natively on **{RESOLVED_DEVICE.upper()}**."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A cinematic macro portrait of a mechanical dragonfly...",
                    lines=3,
                )
                uncond_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, bad anatomy, worst quality, low quality, artifacts, watermark",
                    lines=2,
                )
                input_image = gr.Image(label="Input Image (Optional for Img2Img)", type="pil")

                with gr.Accordion("Sampling Controls", open=False):
                    strength = gr.Slider(0.01, 1.0, value=0.8, step=0.01, label="Denoising Strength (Img2Img)")
                    cfg_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Guidance Scale")
                    steps = gr.Slider(10, 100, value=50, step=1, label="Inference Timesteps (DDPM)")
                    seed = gr.Number(label="Random Seed (-1 for stochastic)", value=-1, precision=0)

                submit_btn = gr.Button("Generate Latents & Render", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Synthesized Output Image")

        submit_btn.click(
            fn=predict,
            inputs=[prompt, uncond_prompt, input_image, strength, cfg_scale, steps, seed],
            outputs=[output_image],
        )

    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_ui()