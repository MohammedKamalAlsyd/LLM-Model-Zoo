# Copyright 2024-2025 The Alibaba Wan Team Authors and Project Contributors.
import gc
import logging
import math
import os
import random
import subprocess
import sys
import time
from typing import Any, Optional, Tuple, Union, cast

# Enable PyTorch expandable segments to prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import diffusers
import gradio as gr
import imageio
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from Zoo.Wan21.utils.model_loader import WanModelContainer, load_wan_submodels

# Safely extract AudioLDMPipeline to satisfy static analyzers
AudioLDMPipeline = getattr(diffusers, "AudioLDMPipeline", None)

# -----------------------------------------------------------------------------
# CUDA & cuDNN Global Optimization Configuration
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    # cuDNN MUST be enabled for 3D convolutions on CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # Avoid algorithm search overhead on dynamic shapes
    
    # Configure PyTorch 2.x SDPA backends
    major, _ = torch.cuda.get_device_capability()
    torch.backends.cuda.enable_flash_sdp(major >= 8)       # FlashAttention on Ampere+ (sm_80+)
    torch.backends.cuda.enable_mem_efficient_sdp(True)    # High-speed memory-efficient attention on T4
    torch.backends.cuda.enable_math_sdp(True)             # Fallback math kernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WanGradioServer")

# High-impact English negative prompt derived from UMT5 multimodal embeddings
DEFAULT_NEG_PROMPT_EN = (
    "bright colors, overexposed, static, blurry details, subtitles, artwork, "
    "painting, still frame, washed out, worst quality, low quality, JPEG artifacts, "
    "ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, "
    "deformed, disfigured, malformed limbs, fused fingers, motionless, cluttered background, "
    "three legs, crowded background, walking backwards"
)

DEFAULT_T2V_PROMPT = (
    "A majestic bald eagle soaring over snow-capped mountain peaks during sunset, "
    "golden hour lighting, cinematic 4k, mountain winds blowing through the canyon"
)

DEFAULT_I2V_PROMPT = (
    "Gentle ocean waves rolling onto a tropical beach, palm trees swaying softly in the breeze, "
    "golden sunset reflections on the water, cinematic slow motion"
)


class KaggleMemoryManager:
    """Utilities to strictly enforce memory limits on 16GB GPUs."""

    @staticmethod
    def flush():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def report_vram(tag: str = ""):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            res = torch.cuda.memory_reserved() / (1024**2)
            logger.info(f"[{tag}] VRAM Allocated: {alloc:.1f}MB | Reserved: {res:.1f}MB")


def sync_vae_device(vae: Any, target_device: Union[torch.device, str], dtype: torch.dtype = torch.float32):
    """
    Synchronizes the entire WanVAE state (model, device attribute, mean, std, scale)
    to a single target device and dtype to prevent cross-device mismatches.
    """
    device_obj = torch.device(target_device)
    vae.device = device_obj
    vae.dtype = dtype

    if hasattr(vae, "model") and vae.model is not None:
        vae.model = vae.model.to(device=device_obj, dtype=dtype)
    if hasattr(vae, "mean") and isinstance(vae.mean, torch.Tensor):
        vae.mean = vae.mean.to(device=device_obj, dtype=dtype)
    if hasattr(vae, "std") and isinstance(vae.std, torch.Tensor):
        vae.std = vae.std.to(device=device_obj, dtype=dtype)
    
    # Rebuild scale tuple on target device
    if hasattr(vae, "mean") and hasattr(vae, "std"):
        vae.scale = [vae.mean, 1.0 / vae.std]
    elif hasattr(vae, "scale") and isinstance(vae.scale, (list, tuple)):
        vae.scale = [
            s.to(device=device_obj, dtype=dtype) if isinstance(s, torch.Tensor) else s
            for s in vae.scale
        ]


class WanAudioEngine:
    """Lightweight Audio/Foley synthesis engine to generate audio tracks for videos."""

    def __init__(self, device: torch.device):
        self.device = device
        self.pipe: Any = None
        self._load_failed: bool = False

    def _lazy_load(self):
        if self.pipe is None and not self._load_failed:
            if AudioLDMPipeline is None:
                logger.warning("AudioLDMPipeline is not installed in current diffusers package.")
                self._load_failed = True
                return

            try:
                logger.info("Loading AudioLDM pipeline for sound effect generation...")
                self.pipe = AudioLDMPipeline.from_pretrained(
                    "cvssp/audioldm-s-full-v2",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                self.pipe = self.pipe.to(self.device)
            except Exception as e:
                logger.error(f"Failed to initialize AudioLDM: {e}")
                self._load_failed = True
                self.pipe = None

    def generate_sound(self, prompt: str, duration_sec: float, output_wav_path: str) -> bool:
        self._lazy_load()
        if self.pipe is None or self._load_failed:
            logger.warning("Audio generator is unavailable. Skipping sound synthesis.")
            return False

        try:
            logger.info(f"Generating {duration_sec:.2f}s audio for prompt: '{prompt[:60]}...'")
            result = self.pipe(
                prompt=prompt,
                negative_prompt="low quality, noise, distortion, static, speaking, talking, speech",
                num_inference_steps=20,
                audio_length_in_s=max(1.0, duration_sec),
            )
            
            audio = result.audios[0] if hasattr(result, "audios") else result[0][0]

            # Convert to 16-bit PCM WAV
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wavfile.write(output_wav_path, rate=16000, data=audio_int16)
            return True
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return False


def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Muxes an audio track into an MP4 video file using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info(f"Audio muxing complete: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"FFmpeg muxing failed ({e}). Returning silent video.")
        return video_path


class WanGradioPipeline:

    def __init__(self, container: WanModelContainer):
        self.c = container
        self.device = torch.device(self.c.device) if isinstance(self.c.device, str) else self.c.device
        self.audio_engine = WanAudioEngine(self.device)
        
        # Hardware-aware precision assignment:
        # T4 (sm_75) and P100 (sm_60) do not have BF16 Tensor Cores.
        # Running BF16 causes slow software emulation (5x-10x slower) and cuDNN conv3d rejections.
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                self.dtype = torch.float16
                logger.info(f"Hardware compute capability is {major}.x (T4/P100). Activating FP16 Tensor Cores.")
            else:
                self.dtype = getattr(self.c, "param_dtype", torch.bfloat16)
                logger.info(f"Hardware compute capability is {major}.x (Ampere/Ada/Hopper). Using BF16.")
        else:
            self.dtype = torch.float32

        self.c.param_dtype = self.dtype

    def _prepare_canvas(self, height: int, width: int) -> Tuple[int, int, int, int]:
        dh = self.c.vae_stride[1] * self.c.patch_size[1]
        dw = self.c.vae_stride[2] * self.c.patch_size[2]
        lat_h = round(height / dh) * self.c.patch_size[1]
        lat_w = round(width / dw) * self.c.patch_size[2]
        return lat_h, lat_w, lat_h * self.c.vae_stride[1], lat_w * self.c.vae_stride[2]

    @torch.no_grad()
    def generate(
        self,
        task: str = "t2v",
        prompt: str = "",
        negative_prompt: str = DEFAULT_NEG_PROMPT_EN,
        image: Optional[Image.Image] = None,
        first_frame: Optional[Image.Image] = None,
        last_frame: Optional[Image.Image] = None,
        width: int = 832,
        height: int = 480,
        frame_num: int = 33,
        steps: int = 20,
        guide_scale: float = 5.0,
        shift: float = 3.0,
        seed: int = -1,
        enable_audio: bool = True,
        progress: Optional[gr.Progress] = None,
    ) -> str:
        KaggleMemoryManager.flush()
        KaggleMemoryManager.report_vram("Before Generation")

        # Fallback to default testing prompt if empty
        if not prompt or prompt.strip() == "":
            prompt = DEFAULT_T2V_PROMPT if task == "t2v" else DEFAULT_I2V_PROMPT
            logger.info(f"Using default testing prompt: {prompt}")

        lat_h, lat_w, ren_h, ren_w = self._prepare_canvas(height, width)
        F_lat = (frame_num - 1) // self.c.vae_stride[0] + 1
        seq_len = math.ceil((lat_h * lat_w) / (self.c.patch_size[1] * self.c.patch_size[2]) * F_lat)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        
        # Generator device matches target tensor device to avoid PyTorch device mismatch
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # ---------------------------------------------------------------------
        # 1. Linguistic Encoding on CPU RAM (Zero GPU VRAM used)
        # ---------------------------------------------------------------------
        logger.info("Encoding text prompts in System RAM (CPU)...")
        context = self.c.text_encoder([prompt], device=torch.device("cpu"))
        context_null = self.c.text_encoder([negative_prompt], device=torch.device("cpu"))

        # Transfer only the token embeddings to GPU
        context = [t.to(device=self.device, dtype=self.dtype) for t in context]
        context_null = [t.to(device=self.device, dtype=self.dtype) for t in context_null]

        # ---------------------------------------------------------------------
        # 2. Image Conditioning (I2V / FLF2V)
        # ---------------------------------------------------------------------
        clip_fea = None
        y = None

        if task == "i2v" and image is not None:
            if self.c.clip is None:
                raise ValueError("CLIP model is required for Image-to-Video but was not found in container.")

            logger.info("Processing conditioning image...")
            self.c.clip.model.to(self.device)
            img_t = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device)
            clip_fea = self.c.clip.visual([img_t[:, None, :, :]])

            self.c.clip.model.to("cpu")
            KaggleMemoryManager.flush()

            # Fully synchronize VAE to GPU in float32 for clean encoding
            sync_vae_device(self.c.vae, self.device, dtype=torch.float32)

            with torch.amp.autocast("cuda", enabled=False):
                img_scaled = F.interpolate(img_t.unsqueeze(0), size=(ren_h, ren_w), mode="bicubic")
                cond_video = torch.cat([
                    img_scaled.transpose(1, 2),
                    torch.zeros(1, 3, frame_num - 1, ren_h, ren_w, device=self.device)
                ], dim=2).squeeze(0).to(dtype=torch.float32)

                y_img = self.c.vae.encode([cond_video])[0].to(dtype=self.dtype)

            sync_vae_device(self.c.vae, "cpu", dtype=torch.float32)
            KaggleMemoryManager.flush()

            msk = torch.zeros(1, frame_num, lat_h, lat_w, device=self.device, dtype=self.dtype)
            msk[:, 0] = 1.0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
            y = [torch.cat([msk, y_img], dim=0)]

        # ---------------------------------------------------------------------
        # 3. Flow Matching Noise Setup
        # ---------------------------------------------------------------------
        noise = torch.randn(
            16, F_lat, lat_h, lat_w,
            dtype=self.dtype,
            generator=generator,
            device=self.device
        )

        self.c.scheduler.set_timesteps(num_inference_steps=steps, device=self.device, shift=shift)
        timesteps = self.c.scheduler.timesteps

        arg_cond = {"context": context, "seq_len": seq_len}
        arg_uncond = {"context": context_null, "seq_len": seq_len}
        if y is not None:
            arg_cond["y"] = y
            arg_uncond["y"] = y
        if clip_fea is not None:
            arg_cond["clip_fea"] = clip_fea
            arg_uncond["clip_fea"] = clip_fea

        # ---------------------------------------------------------------------
        # 4. DiT Sampling Loop (Kept on GPU throughout the loop)
        # ---------------------------------------------------------------------
        logger.info(f"Loading DiT backbone ({self.dtype}) to GPU for denoising...")
        cast(torch.nn.Module, self.c.dit).to(device=self.device, dtype=self.dtype)
        KaggleMemoryManager.report_vram("DiT Active")

        latent = noise
        with torch.amp.autocast("cuda", dtype=self.dtype):
            for t in tqdm(timesteps, desc="Sampling Video Frames"):
                latent_input = [latent]
                t_tensor = torch.tensor([t], device=self.device)

                # Classifier-Free Guidance (CFG) evaluations
                v_cond = self.c.dit(latent_input, t=t_tensor, **arg_cond)[0]
                v_uncond = self.c.dit(latent_input, t=t_tensor, **arg_uncond)[0]
                v_guided = v_uncond + guide_scale * (v_cond - v_uncond)

                # High-order ODE multistep step
                latent = self.c.scheduler.step(
                    model_output=v_guided,
                    timestep=t,
                    sample=latent,
                    return_dict=False,
                )[0]

        logger.info("Denoising complete. Unloading DiT from GPU...")
        cast(torch.nn.Module, self.c.dit).to("cpu")
        KaggleMemoryManager.flush()
        KaggleMemoryManager.report_vram("DiT Offloaded")

        # ---------------------------------------------------------------------
        # 5. VAE Latent Decode (All VAE tensors synchronized to GPU in FP32)
        # ---------------------------------------------------------------------
        logger.info("Loading VAE to GPU for temporal decoding in FP32...")
        
        sync_vae_device(self.c.vae, self.device, dtype=torch.float32)

        with torch.amp.autocast("cuda", enabled=False):
            latent_f32 = latent.to(device=self.device, dtype=torch.float32)
            video = self.c.vae.decode([latent_f32])[0]

        sync_vae_device(self.c.vae, "cpu", dtype=torch.float32)
        KaggleMemoryManager.flush()

        # Render silent video file
        timestamp = int(time.time())
        silent_video_path = f"wan_silent_{timestamp}.mp4"
        self._render_mp4(video, silent_video_path, fps=16)

        # ---------------------------------------------------------------------
        # 6. Optional Sound Effect Synthesis & Multiplexing
        # ---------------------------------------------------------------------
        if enable_audio:
            duration = frame_num / 16.0
            wav_path = f"wan_audio_{timestamp}.wav"
            audio_success = self.audio_engine.generate_sound(prompt, duration, wav_path)

            if audio_success and os.path.exists(wav_path):
                final_video_path = f"wan_video_with_audio_{timestamp}.mp4"
                final_path = mux_audio_video(silent_video_path, wav_path, final_video_path)
                
                # Cleanup intermediate WAV file
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

                return final_path

        return silent_video_path

    def _render_mp4(self, tensor: torch.Tensor, output_path: str, fps: int = 16):
        scaled_tensor = tensor.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255.0).byte()
        frames_np = scaled_tensor.permute(1, 2, 3, 0).cpu().numpy()
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        logger.info(f"Rendered video saved: {output_path}")


# ============================================================================
# Gradio Web Interface Construction
# ============================================================================

pipeline_instance: Optional[WanGradioPipeline] = None


def init_pipeline(model_scale: str):
    global pipeline_instance
    KaggleMemoryManager.flush()
    container = load_wan_submodels(
        task="t2v",
        scale=model_scale,
        offload_model=True,
        t5_cpu=True,
    )
    pipeline_instance = WanGradioPipeline(container)
    return f"Model successfully loaded: {model_scale} (Optimized for Kaggle GPU, T5 on CPU)"


def run_t2v(prompt, neg_prompt, resolution, frames, steps, cfg, shift, seed, enable_audio):
    global pipeline_instance
    if pipeline_instance is None:
        init_pipeline("1.3B")
    assert pipeline_instance is not None, "Pipeline failed to initialize"
    w, h = [int(x) for x in resolution.split("x")]
    return pipeline_instance.generate(
        task="t2v",
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=w,
        height=h,
        frame_num=int(frames),
        steps=int(steps),
        guide_scale=float(cfg),
        shift=float(shift),
        seed=int(seed),
        enable_audio=bool(enable_audio),
    )


def run_i2v(image, prompt, neg_prompt, resolution, frames, steps, cfg, shift, seed, enable_audio):
    global pipeline_instance
    if pipeline_instance is None:
        init_pipeline("1.3B")
    assert pipeline_instance is not None, "Pipeline failed to initialize"
    w, h = [int(x) for x in resolution.split("x")]
    return pipeline_instance.generate(
        task="i2v",
        prompt=prompt,
        negative_prompt=neg_prompt,
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


def build_app() -> gr.Blocks:
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
            # Wan2.1: Unified Video Studio with Audio (Kaggle T4 / P100 Optimized)
            Generate spatio-temporally coherent videos using the **Wan2.1** Continuous Flow-Matching DiT architecture.
            *Includes synchronized AI sound effect generation & audio-video muxing.*
            """
        )

        with gr.Row():
            scale_dropdown = gr.Dropdown(
                label="Model Size",
                choices=["1.3B", "14B"],
                value="1.3B",
                info="1.3B is strictly recommended for Kaggle T4 (16GB VRAM).",
            )
            load_status = gr.Textbox(label="System Status", value="Ready to initialize", interactive=False)
            init_btn = gr.Button("Initialize / Switch Model", variant="secondary")

        init_btn.click(fn=init_pipeline, inputs=[scale_dropdown], outputs=[load_status])

        with gr.Tabs():
            # Tab 1: Text to Video
            with gr.TabItem("Text to Video (T2V)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            value=DEFAULT_T2V_PROMPT,
                            lines=3,
                        )
                        t2v_neg_prompt = gr.Textbox(
                            label="Negative Prompt (English)",
                            value=DEFAULT_NEG_PROMPT_EN,
                            lines=2,
                        )
                        with gr.Row():
                            t2v_res = gr.Dropdown(
                                label="Resolution",
                                choices=["832x480", "480x832", "1280x720", "720x1280"],
                                value="832x480",
                                info="480p is recommended for fast T4 rendering.",
                            )
                            t2v_frames = gr.Dropdown(
                                label="Frames (4n+1)",
                                choices=[17, 33, 49, 81],
                                value=33,
                                info="33 frames (~2 sec) renders in ~2-3 minutes.",
                            )
                        t2v_audio = gr.Checkbox(
                            label="Generate AI Sound Effects (Audio)",
                            value=True,
                            info="Synthesizes sound effects matching the prompt and synchronizes to video duration.",
                        )
                        with gr.Accordion("Advanced Sampling Parameters", open=False):
                            t2v_steps = gr.Slider(label="Sampling Steps", minimum=15, maximum=50, value=20, step=1)
                            t2v_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.5)
                            t2v_shift = gr.Slider(label="Flow Shift Factor", minimum=1.0, maximum=8.0, value=3.0, step=0.5)
                            t2v_seed = gr.Number(label="Seed (-1 for random)", value=-1)

                        t2v_btn = gr.Button("Generate Video", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=5):
                        t2v_output = gr.Video(label="Rendered Video (with Audio)", autoplay=True)

                t2v_btn.click(
                    fn=run_t2v,
                    inputs=[t2v_prompt, t2v_neg_prompt, t2v_res, t2v_frames, t2v_steps, t2v_cfg, t2v_shift, t2v_seed, t2v_audio],
                    outputs=[t2v_output],
                )

            # Tab 2: Image to Video
            with gr.TabItem("Image to Video (I2V)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        i2v_image = gr.Image(label="Input Anchor Frame", type="pil")
                        i2v_prompt = gr.Textbox(
                            label="Motion Prompt",
                            value=DEFAULT_I2V_PROMPT,
                            lines=2,
                        )
                        i2v_neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEG_PROMPT_EN, lines=2)
                        with gr.Row():
                            i2v_res = gr.Dropdown(label="Resolution", choices=["832x480", "480x832"], value="832x480")
                            i2v_frames = gr.Dropdown(label="Frames", choices=[17, 33, 49, 81], value=33)
                        i2v_audio = gr.Checkbox(
                            label="Generate AI Sound Effects (Audio)",
                            value=True,
                            info="Synthesizes sound effects matching the scene.",
                        )
                        with gr.Accordion("Advanced Parameters", open=False):
                            i2v_steps = gr.Slider(label="Steps", minimum=15, maximum=50, value=20, step=1)
                            i2v_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                            i2v_shift = gr.Slider(label="Shift Factor", minimum=1.0, maximum=6.0, value=3.0, step=0.5)
                            i2v_seed = gr.Number(label="Seed", value=-1)

                        i2v_btn = gr.Button("Animate Image", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=5):
                        i2v_output = gr.Video(label="Animated Output (with Audio)", autoplay=True)

                i2v_btn.click(
                    fn=run_i2v,
                    inputs=[i2v_image, i2v_prompt, i2v_neg_prompt, i2v_res, i2v_frames, i2v_steps, i2v_cfg, i2v_shift, i2v_seed, i2v_audio],
                    outputs=[i2v_output],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    # Share=True creates a public gradio.live link accessible outside Kaggle
    app.queue(max_size=3).launch(share=True, server_port=7860)