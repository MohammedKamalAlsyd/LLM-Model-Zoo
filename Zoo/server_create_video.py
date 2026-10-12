"""Universal Video Studio Server & CLI Runner for Wan2.1.

Supports interactive Gradio web serving and direct headless command-line execution
with FoleyCrafter audio co-synthesis and aggressive VRAM purging via Zoo.Common.model_loader.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Setup workspace paths
current_dir = Path(__file__).resolve().parent
zoo_dir = current_dir.parent
root_dir = zoo_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gradio as gr
from PIL import Image
import torch

from Zoo.Common.model_loader import auto_detect_device_and_dtype, purge_memory
from Zoo.Wan21.Wan21 import Wan21

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Wan21Studio")

# Global singleton pipeline holder for Web UI
WAN_INSTANCE: Optional[Wan21] = None


def get_pipeline(task: str = "t2v", scale: str = "1.3B", resolution: str = "480P") -> Wan21:
    """Lazy loader reusing or instantiating the requested Wan2.1 model."""
    global WAN_INSTANCE
    if WAN_INSTANCE is not None:
        if WAN_INSTANCE.config.task == task and WAN_INSTANCE.config.scale == scale:
            return WAN_INSTANCE
        logger.info("[*] Swapping active Wan2.1 model in VRAM...")
        del WAN_INSTANCE
        purge_memory()

    logger.info(f"[*] Initializing Wan2.1 [{task.upper()} - {scale}]...")
    WAN_INSTANCE = Wan21.from_pretrained(
        task=task,
        scale=scale,
        resolution=resolution,
        offload_model=True,
    )
    return WAN_INSTANCE


def run_generation(
    task: str,
    prompt: str,
    neg_prompt: str,
    audio_prompt: str,
    resolution: str,
    frames: int,
    steps: int,
    cfg: float,
    shift: float,
    seed: int,
    enable_audio: bool,
    image: Optional[Image.Image] = None,
) -> str:
    w, h = [int(x) for x in resolution.split("x")]
    res_tag = "720P" if max(w, h) >= 1280 else "480P"
    pipeline = get_pipeline(task=task, scale="1.3B", resolution=res_tag)

    return pipeline.generate(
        prompt=prompt,
        negative_prompt=neg_prompt,
        audio_prompt=audio_prompt,
        image=image,
        width=w,
        height=h,
        frame_num=int(frames),
        steps=int(steps),
        guide_scale=float(cfg),
        shift=float(shift),
        seed=int(seed),
        enable_audio=bool(enable_audio),
    )


def build_ui() -> gr.Blocks:
    custom_css = """
    <style>
    .gradio-container {max-width: 1100px !important; margin: 0 auto !important;}
    .generate-btn {background: #ff5722 !important; color: white !important; font-size: 16px !important;}
    </style>
    """

    with gr.Blocks(title="Wan2.1 Unified Video Studio") as demo:
        gr.HTML(custom_css)
        gr.Markdown(
            """
            # 🎬 Wan2.1: Continuous Flow Video Studio
            Generate high-fidelity videos using **Wan2.1 3D DiT** paired with **FoleyCrafter** video-to-audio co-generation.
            """
        )

        with gr.Tabs():
            # -------------------------------------------------------------
            # TAB 1: Text-to-Video
            # -------------------------------------------------------------
            with gr.TabItem("Text to Video (T2V)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        t2v_prompt = gr.Textbox(
                            label="Video Prompt",
                            value="A cinematic shot of a red vintage convertible driving along the Pacific Coast Highway at sunset.",
                            lines=3,
                        )
                        t2v_audio_prompt = gr.Textbox(
                            label="Audio / Foley Prompt (Optional)",
                            placeholder="Leave blank to automatically derive from visual prompt.",
                            lines=2,
                        )
                        t2v_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="bright colors, static, blurry details, washed out, low quality, deformed, motionless",
                            lines=2,
                        )
                        with gr.Row():
                            t2v_res = gr.Dropdown(
                                label="Resolution",
                                choices=["832x480", "480x832", "1280x720", "720x1280"],
                                value="832x480",
                            )
                            t2v_frames = gr.Dropdown(label="Frames (4n+1)", choices=[17, 33, 49, 81], value=33)

                        t2v_audio = gr.Checkbox(label="Synthesize Motion-Synchronized Foley Audio", value=True)

                        with gr.Accordion("Advanced Sampling Parameters", open=False):
                            t2v_steps = gr.Slider(label="Sampling Steps", minimum=15, maximum=50, value=20, step=1)
                            t2v_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.5)
                            t2v_shift = gr.Slider(label="Flow Shift", minimum=1.0, maximum=8.0, value=3.0, step=0.5)
                            t2v_seed = gr.Number(label="Seed (-1 for random)", value=-1)

                        t2v_btn = gr.Button("Generate Video with Sound", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=5):
                        t2v_output = gr.Video(label="Rendered Video Output", autoplay=True)

                t2v_btn.click(
                    fn=lambda p, np, ap, r, f, st, c, sh, se, a: run_generation(
                        "t2v", p, np, ap, r, f, st, c, sh, se, a, image=None
                    ),
                    inputs=[t2v_prompt, t2v_neg_prompt, t2v_audio_prompt, t2v_res, t2v_frames, t2v_steps, t2v_cfg, t2v_shift, t2v_seed, t2v_audio],
                    outputs=[t2v_output],
                )

            # -------------------------------------------------------------
            # TAB 2: Image-to-Video
            # -------------------------------------------------------------
            with gr.TabItem("Image to Video (I2V)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        i2v_img = gr.Image(label="Anchor Image", type="pil")
                        i2v_prompt = gr.Textbox(
                            label="Motion Prompt",
                            value="Gentle camera zoom in, ocean waves softly crashing on the sandy shore.",
                            lines=2,
                        )
                        i2v_audio_prompt = gr.Textbox(
                            label="Foley Sound Prompt (Optional)",
                            placeholder="Describe sound effects for this scene.",
                            lines=2,
                        )
                        i2v_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="static, blurry, jittery motion, deformed, low quality",
                            lines=2,
                        )
                        with gr.Row():
                            i2v_res = gr.Dropdown(label="Resolution", choices=["832x480", "480x832"], value="832x480")
                            i2v_frames = gr.Dropdown(label="Frames", choices=[17, 33, 49, 81], value=33)

                        i2v_audio = gr.Checkbox(label="Synthesize Synchronized Foley Audio", value=True)

                        with gr.Accordion("Advanced Parameters", open=False):
                            i2v_steps = gr.Slider(label="Steps", minimum=15, maximum=50, value=20, step=1)
                            i2v_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                            i2v_shift = gr.Slider(label="Shift", minimum=1.0, maximum=6.0, value=3.0, step=0.5)
                            i2v_seed = gr.Number(label="Seed", value=-1)

                        i2v_btn = gr.Button("Animate Image with Sound", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=5):
                        i2v_output = gr.Video(label="Rendered Video Output", autoplay=True)

                i2v_btn.click(
                    fn=lambda img, p, np, ap, r, f, st, c, sh, se, a: run_generation(
                        "i2v", p, np, ap, r, f, st, c, sh, se, a, image=img
                    ),
                    inputs=[i2v_img, i2v_prompt, i2v_neg_prompt, i2v_audio_prompt, i2v_res, i2v_frames, i2v_steps, i2v_cfg, i2v_shift, i2v_seed, i2v_audio],
                    outputs=[i2v_output],
                )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.1 Unified Studio Server & Headless Generator")
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v"], help="Generation task")
    parser.add_argument("--scale", type=str, default="1.3B", choices=["1.3B", "14B"], help="Model parameter scale")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for direct CLI generation")
    parser.add_argument("--image", type=str, default=None, help="Input image path for I2V")
    parser.add_argument("--resolution", type=str, default="832x480", help="Output resolution (WxH)")
    parser.add_argument("--frames", type=int, default=33, help="Total frames (must be 4n+1)")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--no_audio", action="store_true", help="Disable Foley audio synthesis")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio tunnel")
    args = parser.parse_args()

    # Headless CLI Execution Mode
    if args.prompt is not None:
        logger.info(f"[*] Running direct headless CLI generation for task: {args.task}...")
        img = Image.open(args.image).convert("RGB") if args.image else None
        res_tag = "720P" if "1280" in args.resolution else "480P"
        pipeline = Wan21.from_pretrained(task=args.task, scale=args.scale, resolution=res_tag, offload_model=True)

        w, h = [int(x) for x in args.resolution.split("x")]
        out_path = pipeline.generate(
            prompt=args.prompt,
            image=img,
            width=w,
            height=h,
            frame_num=args.frames,
            steps=args.steps,
            guide_scale=args.cfg,
            enable_audio=not args.no_audio,
        )
        print(f"\n[✓] Video successfully rendered to: {out_path}")
        return

    # Web Studio Mode
    logger.info("[*] Launching Wan2.1 Gradio Web Studio...")
    app = build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()