"""Unified Multi-Model Image Generation Studio with Active VRAM/RAM Swapping."""

from typing import Optional, Union
import gradio as gr
from PIL import Image

from Zoo.Common.model_loader import auto_detect_device_and_dtype, purge_memory
from Zoo.FLUX1Schnell.FLUX import FluxModel
from Zoo.FLUX1Schnell.pipeline.FluxPipeline import FluxPipeline
from Zoo.StableDiffusion.StableDiffusion import StableDiffusionModel
from Zoo.StableDiffusion.pipeline.StableDiffusionPipeline import StableDiffusionPipeline

ACTIVE_MODEL_NAME: Optional[str] = None
CURRENT_PIPELINE: Union[StableDiffusionPipeline, FluxPipeline, None] = None

PRESET_PROMPTS = {
    "Custom": {
        "prompt": "",
        "negative_prompt": "blurry, bad anatomy, low quality, artifacts",
    },
    "Cinematic Portrait": {
        "prompt": "A cinematic dramatic portrait of an elderly sea captain with a weathered face, wearing a yellow raincoat, moody side lighting, highly detailed, 8k resolution",
        "negative_prompt": "blurry, low quality, distorted face, extra limbs, bad anatomy, flat lighting, oversaturated",
    },
    "Cyberpunk Cityscape": {
        "prompt": "A futuristic cyberpunk metropolis at night, vibrant neon signs, towering skyscrapers, wet asphalt reflecting holographic advertisements, ultra-detailed 8k",
        "negative_prompt": "blurry, low quality, oversaturated, messy, bad composition, noisy, pixelated",
    },
    "Fantasy Forest": {
        "prompt": "Enchanted ancient forest with glowing flora, mystical atmosphere, mossy stone ruins, ethereal sunlight filtering through canopy, digital art style",
        "negative_prompt": "blurry, bad quality, realistic photo, text, watermark, signature, ugly",
    },
    "Anime / Studio Ghibli Style": {
        "prompt": "A serene countryside cottage surrounded by vibrant wildflower fields, soft fluffy summer clouds in a bright blue sky, Studio Ghibli style, detailed anime background",
        "negative_prompt": "blurry, 3d render, photo, photorealistic, ugly, dark, depressing, low effort",
    },
    "Photorealistic Product": {
        "prompt": "Studio product photograph of a luxury glass perfume bottle resting on a reflective dark marble surface, elegant diffuse lighting, ultra-sharp focus",
        "negative_prompt": "blurry, low quality, dust, scratches, glare, bad reflections, distortion",
    },
}


def get_pipeline(model_choice: str) -> Union[StableDiffusionPipeline, FluxPipeline]:
    """Dynamically loads the requested pipeline while evicting the inactive model to prevent OOM."""
    global ACTIVE_MODEL_NAME, CURRENT_PIPELINE

    if ACTIVE_MODEL_NAME == model_choice and CURRENT_PIPELINE is not None:
        return CURRENT_PIPELINE

    # Evict current pipeline from RAM and GPU VRAM
    if CURRENT_PIPELINE is not None:
        print(f"\n[Studio] Evicting '{ACTIVE_MODEL_NAME}' from memory...")
        del CURRENT_PIPELINE
        CURRENT_PIPELINE = None
        purge_memory()

    device, dtype = auto_detect_device_and_dtype()

    if model_choice == "Stable Diffusion v1.5":
        print(f"[Studio] Loading Stable Diffusion on {device.upper()} ({dtype})...")
        model = StableDiffusionModel.from_pretrained_weights(device=device, dtype=dtype)
        CURRENT_PIPELINE = StableDiffusionPipeline(model=model)
    elif model_choice == "FLUX.1 [schnell]":
        print(f"[Studio] Loading FLUX.1 [schnell] Streaming Pipeline on {device.upper()} ({dtype})...")
        CURRENT_PIPELINE = FluxModel.from_pretrained_weights(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    ACTIVE_MODEL_NAME = model_choice
    return CURRENT_PIPELINE


def generate_image_dispatch(
    model_choice: str,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    seed: int,
) -> Image.Image:
    """Dispatches the generation request to the active pipeline."""
    active_seed = None if int(seed) == -1 else int(seed)
    pipe = get_pipeline(model_choice)

    if isinstance(pipe, StableDiffusionPipeline):
        return pipe(
            prompt=prompt,
            uncond_prompt=negative_prompt,
            num_inference_steps=int(steps),
            cfg_scale=float(cfg_scale),
            height=int(height),
            width=int(width),
            seed=active_seed,
        )
    elif isinstance(pipe, FluxPipeline):
        return pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            seed=active_seed,
        )
    raise RuntimeError("Pipeline failed to initialize.")


def launch_unified_studio() -> None:
    dev, dtype = auto_detect_device_and_dtype()

    with gr.Blocks(title="Unified Image Generation Studio") as demo:
        gr.Markdown(
            f"# 🎨 Unified Image Generation Studio\n"
            f"Execution running on **{dev.upper()}** ({dtype}). Models are swapped dynamically to maintain a **< 2 GB VRAM** profile."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Radio(
                    choices=["FLUX.1 [schnell]", "Stable Diffusion v1.5"],
                    value="FLUX.1 [schnell]",
                    label="Generative Architecture",
                )
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_PROMPTS.keys()),
                    value="Custom",
                    label="Preset Prompts (Select a preset or customize freely)",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your image or choose a preset above...",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (SD v1.5 only)",
                    value="blurry, bad anatomy, low quality, artifacts",
                    lines=2,
                    interactive=False,
                )

                with gr.Row():
                    height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                    width = gr.Slider(512, 1536, value=1024, step=64, label="Width")

                with gr.Row():
                    steps = gr.Slider(1, 100, value=4, step=1, label="Steps (FLUX: 4, SD: 50)")
                    cfg_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Scale (SD only)", interactive=False)

                seed = gr.Number(label="Random Seed (-1 for stochastic)", value=-1, precision=0)
                submit_btn = gr.Button("🎨 Synthesize Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Synthesized Output", type="pil")

        def update_preset_fields(preset_name: str):
            preset = PRESET_PROMPTS.get(preset_name, PRESET_PROMPTS["Custom"])
            return preset["prompt"], preset["negative_prompt"]

        def update_ui_defaults(choice: str):
            if choice == "FLUX.1 [schnell]":
                return 1024, 1024, 4, gr.update(interactive=False), gr.update(interactive=False)
            else:
                return 512, 512, 50, gr.update(interactive=True), gr.update(interactive=True)

        preset_dropdown.change(
            fn=update_preset_fields,
            inputs=[preset_dropdown],
            outputs=[prompt, negative_prompt],
        )

        model_choice.change(
            fn=update_ui_defaults,
            inputs=[model_choice],
            outputs=[height, width, steps, negative_prompt, cfg_scale],
        )

        submit_btn.click(
            fn=generate_image_dispatch,
            inputs=[model_choice, prompt, negative_prompt, height, width, steps, cfg_scale, seed],
            outputs=[output_image],
        )

    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_unified_studio()