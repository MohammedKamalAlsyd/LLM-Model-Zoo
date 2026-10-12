"""Master Unified Video Generation Model for Wan2.1.

Encapsulates 3D DiT, 3D Causal VAE, UMT5-XXL text encoder, XLM-RoBERTa CLIP,
and continuous Flow-Matching schedulers with automatic VRAM management.
"""

import gc
import importlib
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import diffusers
from huggingface_hub import snapshot_download
import imageio
import numpy as np
from PIL import Image
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from Zoo.Common.model_loader import auto_detect_device_and_dtype, purge_memory
from Zoo.Wan21.configs import Wan21Config
from Zoo.Wan21.modules.CLIP import CLIPModel
from Zoo.Wan21.modules.DiT import VaceWanModel, WanModel
from Zoo.Wan21.modules.FlowScheduler import FlowUniPCMultistepScheduler, WanFlowScheduler
from Zoo.Wan21.modules.T5 import T5EncoderModel
from Zoo.Wan21.modules.VAE import WanVAE

logger = logging.getLogger("Wan21")

HF_REPO_MAP = {
    "t2v-1.3B": "Wan-AI/Wan2.1-T2V-1.3B",
    "t2v-14B": "Wan-AI/Wan2.1-T2V-14B",
    "i2v-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P",
    "i2v-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P",
    "flf2v-14B": "Wan-AI/Wan2.1-I2V-14B-720P",
    "vace-1.3B": "Wan-AI/Wan2.1-T2V-1.3B",
    "vace-14B": "Wan-AI/Wan2.1-T2V-14B",
}

AudioLDM2Pipeline = getattr(diffusers, "AudioLDM2Pipeline", None)
AudioLDMPipeline = getattr(diffusers, "AudioLDMPipeline", None)
DPMSolverMultistepScheduler = getattr(diffusers, "DPMSolverMultistepScheduler", None)


class FoleyCrafterAudioEngine:
    """Video-to-Audio (V2A) Engine with FoleyCrafter and AudioLDM fallback."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.engine_type: Optional[str] = None
        self.pipe: Any = None
        self._load_failed: bool = False

    def _lazy_load(self) -> None:
        if self.pipe is not None or self._load_failed:
            return

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            fc_module = importlib.import_module("foleycrafter.pipelines.dymctrl_pipeline")
            DymctrlPipeline = getattr(fc_module, "DymctrlPipeline")
            logger.info("Initializing FoleyCrafter ('ymq22/FoleyCrafter')...")
            self.pipe = DymctrlPipeline.from_pretrained("ymq22/FoleyCrafter", torch_dtype=torch_dtype).to(self.device)
            self.engine_type = "foleycrafter"
            return
        except Exception:
            logger.warning("FoleyCrafter unavailable; engaging AudioLDM fallback...")

        try:
            lm = GPT2LMHeadModel.from_pretrained("cvssp/audioldm2", subfolder="language_model", torch_dtype=torch_dtype)
            if AudioLDM2Pipeline is not None:
                self.pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", language_model=lm, torch_dtype=torch_dtype)
                if DPMSolverMultistepScheduler is not None:
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
                    )
                self.pipe = self.pipe.to(self.device)
                self.engine_type = "audioldm2"
                return
        except Exception:
            pass

        try:
            if AudioLDMPipeline is not None:
                self.pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-m-full", torch_dtype=torch_dtype)
                self.pipe = self.pipe.to(self.device)
                self.engine_type = "audioldm"
                return
        except Exception as ex:
            logger.error(f"All audio models failed to initialize: {ex}")
            self._load_failed = True

    def _post_process_audio(self, raw_audio: Any, sample_rate: int, target_duration: float) -> np.ndarray:
        if isinstance(raw_audio, torch.Tensor):
            audio = raw_audio.detach().cpu().float().numpy()
        else:
            audio = np.asarray(raw_audio, dtype=np.float32)

        sos = signal.butter(4, 40, "hp", fs=sample_rate, output="sos")
        filt_result = signal.sosfilt(sos, audio)
        audio = np.asarray(filt_result[0] if isinstance(filt_result, tuple) else filt_result, dtype=np.float32)

        target_samples = int(target_duration * sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        fade_len = int(0.05 * sample_rate)
        if len(audio) > 2 * fade_len:
            fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
            fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
            audio[:fade_len] *= fade_in
            audio[-fade_len:] *= fade_out

        max_val = float(np.max(np.abs(audio)))
        if max_val > 1e-5:
            target_peak = 10.0 ** (-1.0 / 20.0)
            audio = (audio / max_val) * target_peak

        return audio

    def generate_sound(self, video_path: str, audio_prompt: str, duration_sec: float, output_wav_path: str) -> bool:
        self._lazy_load()
        if self.pipe is None or self._load_failed:
            return False

        try:
            if hasattr(self.pipe, "to"):
                self.pipe.to(self.device)
            sample_rate = 16000

            if self.engine_type == "foleycrafter":
                result = self.pipe(prompt=audio_prompt, video_path=video_path, num_inference_steps=35, guidance_scale=7.5)
                raw_audio = result.audios[0] if hasattr(result, "audios") else result[0]
            else:
                result = self.pipe(
                    prompt=audio_prompt, num_inference_steps=40, guidance_scale=5.0, audio_length_in_s=max(5.0, duration_sec)
                )
                raw_audio = result.audios[0] if hasattr(result, "audios") else result[0][0]

            processed_audio = self._post_process_audio(raw_audio, sample_rate, duration_sec)
            audio_int16 = (processed_audio * 32767).clip(-32768, 32767).astype(np.int16)
            wavfile.write(output_wav_path, rate=sample_rate, data=audio_int16)

            if hasattr(self.pipe, "to"):
                self.pipe.to("cpu")
            purge_memory()
            return True
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            if hasattr(self.pipe, "to"):
                self.pipe.to("cpu")
            purge_memory()
            return False


def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Muxes audio and video tracks via ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path
    except Exception as e:
        logger.warning(f"Audio muxing failed: {e}. Returning silent video.")
        return video_path


class Wan21(nn.Module):
    """Top-Level Wan2.1 Multimodal Video Foundation Model Wrapper."""

    def __init__(
        self,
        config: Wan21Config,
        vae: WanVAE,
        text_encoder: T5EncoderModel,
        dit: Union[WanModel, VaceWanModel],
        scheduler: WanFlowScheduler,
        clip: Optional[CLIPModel] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        offload_model: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        resolved_device, resolved_dtype = auto_detect_device_and_dtype(device, dtype)
        self.device = torch.device(resolved_device)
        self.dtype = resolved_dtype
        self.offload_model = offload_model

        self.vae = vae
        self.text_encoder = text_encoder
        self.clip = clip
        self.dit = dit
        self.scheduler = scheduler
        self.audio_engine = FoleyCrafterAudioEngine(self.device)

    @classmethod
    def from_pretrained(
        cls,
        task: str = "t2v",
        scale: str = "1.3B",
        resolution: str = "480P",
        checkpoint_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        offload_model: bool = True,
    ) -> "Wan21":
        """Instantiates and loads pre-trained weights for Wan2.1."""
        resolved_device, resolved_dtype = auto_detect_device_and_dtype(device, dtype)
        dev = torch.device(resolved_device)

        # 1. Download or locate weights
        if checkpoint_dir is None:
            key = f"{task}-{scale}"
            if task == "i2v":
                key = f"{task}-{scale}-{resolution}"
            repo_id = HF_REPO_MAP.get(key, HF_REPO_MAP["t2v-1.3B"])
            checkpoint_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.pth", "*.safetensors", "*.json", "google/umt5-xxl/*"],
            )

        config = Wan21Config.from_preset(task=task, scale=scale, resolution=resolution)

        # 2. VAE (Always stored and decoded in FP32)
        vae_path = os.path.join(checkpoint_dir, "Wan2.1_VAE.pth")
        vae = WanVAE(config=config.vae_config, vae_pth=vae_path, device="cpu" if offload_model else dev, dtype=torch.float32)

        # 3. UMT5-XXL
        t5_path = os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        tok_path = os.path.join(checkpoint_dir, "google/umt5-xxl")
        text_encoder = T5EncoderModel(
            config=config.text_config,
            dtype=torch.bfloat16,
            device="cpu",
            checkpoint_path=t5_path if os.path.exists(t5_path) else None,
            tokenizer_path=tok_path if os.path.exists(tok_path) else "google/umt5-xxl",
        )

        # 4. CLIP (for I2V / FLF2V)
        clip = None
        if task in ("i2v", "flf2v"):
            clip_path = os.path.join(checkpoint_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
            clip = CLIPModel(dtype=torch.float16, device="cpu" if offload_model else dev, checkpoint_path=clip_path)

        # 5. DiT Backbone
        dit: Union[WanModel, VaceWanModel]
        if task == "vace":
            dit = VaceWanModel(**config.dit_config.__dict__)
        else:
            dit = WanModel(**config.dit_config.__dict__)

        for candidate in ["diffusion_pytorch_model.safetensors", f"Wan2.1_T2V_{scale}.pth", f"Wan2.1_I2V_14B_{resolution}.pth"]:
            p = os.path.join(checkpoint_dir, candidate)
            if os.path.exists(p):
                logger.info(f"Loading DiT state from {p}...")
                if p.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state = load_file(p)
                else:
                    state = torch.load(p, map_location="cpu")
                dit.load_state_dict(state, strict=False)
                break

        dit = dit.to(dtype=resolved_dtype)
        if not offload_model:
            dit = dit.to(dev)
        dit.eval().requires_grad_(False)

        # 6. Flow Scheduler
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.scheduler_config.num_train_timesteps,
            shift=config.scheduler_config.shift,
        )

        return cls(
            config=config,
            vae=vae,
            text_encoder=text_encoder,
            dit=dit,
            scheduler=scheduler,
            clip=clip,
            device=dev,
            dtype=resolved_dtype,
            offload_model=offload_model,
        )

    def _prepare_canvas(self, height: int, width: int) -> Tuple[int, int, int, int]:
        dh = self.config.vae_stride[1] * self.config.patch_size[1]
        dw = self.config.vae_stride[2] * self.config.patch_size[2]
        lat_h = round(height / dh) * self.config.patch_size[1]
        lat_w = round(width / dw) * self.config.patch_size[2]
        return lat_h, lat_w, lat_h * self.config.vae_stride[1], lat_w * self.config.vae_stride[2]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "bright colors, static, blurry details, washed out, low quality, deformed, motionless",
        audio_prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        width: int = 832,
        height: int = 480,
        frame_num: int = 33,
        steps: int = 20,
        guide_scale: float = 5.0,
        shift: Optional[float] = None,
        seed: int = -1,
        enable_audio: bool = True,
    ) -> str:
        """Executes full continuous flow-matching sampling loop and audio synthesis."""
        purge_memory()
        lat_h, lat_w, ren_h, ren_w = self._prepare_canvas(height, width)
        F_lat = (frame_num - 1) // self.config.vae_stride[0] + 1
        seq_len = math.ceil((lat_h * lat_w) / (self.config.patch_size[1] * self.config.patch_size[2]) * F_lat)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 1. Text Context Encoding on CPU RAM
        context = self.text_encoder([prompt], device=torch.device("cpu"))
        context_null = self.text_encoder([negative_prompt], device=torch.device("cpu"))
        context = [t.to(device=self.device, dtype=self.dtype) for t in context]
        context_null = [t.to(device=self.device, dtype=self.dtype) for t in context_null]

        # 2. Visual Conditioning (I2V)
        clip_fea = None
        y = None
        if self.config.task == "i2v" and image is not None:
            if self.clip is None:
                raise ValueError("CLIP model must be initialized for Image-to-Video generation.")

            self.clip.model.to(self.device)
            img_t = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device)
            clip_fea = self.clip.visual([img_t[:, None, :, :]])
            self.clip.model.to("cpu")
            purge_memory()

            self.vae.model.to(self.device)
            img_scaled = F.interpolate(img_t.unsqueeze(0), size=(ren_h, ren_w), mode="bicubic")
            cond_video = torch.cat([
                img_scaled.transpose(1, 2),
                torch.zeros(1, 3, frame_num - 1, ren_h, ren_w, device=self.device)
            ], dim=2).squeeze(0).to(dtype=torch.float32)

            y_img = self.vae.encode([cond_video])[0].to(dtype=self.dtype)
            self.vae.model.to("cpu")
            purge_memory()

            msk = torch.zeros(1, frame_num, lat_h, lat_w, device=self.device, dtype=self.dtype)
            msk[:, 0] = 1.0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
            y = [torch.cat([msk, y_img], dim=0)]

        # 3. Latent Flow Initialization
        noise = torch.randn(16, F_lat, lat_h, lat_w, dtype=self.dtype, generator=generator, device=self.device)
        shift_val = shift if shift is not None else self.config.scheduler_config.shift
        self.scheduler.set_timesteps(num_inference_steps=steps, device=self.device, shift=shift_val)
        timesteps = self.scheduler.timesteps

        arg_cond = {"context": context, "seq_len": seq_len}
        arg_uncond = {"context": context_null, "seq_len": seq_len}
        if y is not None:
            arg_cond["y"], arg_uncond["y"] = y, y
        if clip_fea is not None:
            arg_cond["clip_fea"], arg_uncond["clip_fea"] = clip_fea, clip_fea

        # 4. DiT Denoising Loop
        self.dit.to(device=self.device, dtype=self.dtype)
        latent = noise
        with torch.amp.autocast("cuda", dtype=self.dtype):
            for t in tqdm(timesteps, desc="Sampling Video Frames"):
                latent_input = [latent]
                t_tensor = torch.tensor([t], device=self.device)
                v_cond = self.dit(latent_input, t=t_tensor, **arg_cond)[0]
                v_uncond = self.dit(latent_input, t=t_tensor, **arg_uncond)[0]
                v_guided = v_uncond + guide_scale * (v_cond - v_uncond)
                latent = self.scheduler.step(model_output=v_guided, timestep=t, sample=latent, return_dict=False)[0]

        if self.offload_model:
            self.dit.to("cpu")
            purge_memory()

        # 5. Causal VAE Decoding (in FP32)
        self.vae.model.to(self.device)
        latent_f32 = latent.to(device=self.device, dtype=torch.float32)
        video = self.vae.decode([latent_f32])[0]
        if self.offload_model:
            self.vae.model.to("cpu")
            purge_memory()

        # 6. Render Video
        timestamp = int(time.time())
        silent_video = f"wan_{timestamp}_silent.mp4"
        scaled = video.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255.0).byte()
        frames_np = scaled.permute(1, 2, 3, 0).cpu().numpy()
        writer = imageio.get_writer(silent_video, fps=self.config.fps, codec="libx264", quality=8)
        for f in frames_np:
            writer.append_data(f)
        writer.close()

        # 7. FoleyCrafter Audio Synthesis
        if enable_audio:
            if not audio_prompt or audio_prompt.strip() == "":
                cleaned = re.sub(r"(4k|8k|cinematic|photorealistic|sharp focus|masterpiece)", "", prompt, flags=re.I)
                audio_prompt = f"realistic Foley sound effect of {cleaned.strip()}, crisp studio quality recording"

            wav_path = f"wan_{timestamp}_audio.wav"
            duration = frame_num / float(self.config.fps)
            success = self.audio_engine.generate_sound(silent_video, audio_prompt, duration, wav_path)

            if success and os.path.exists(wav_path):
                final_video = f"wan_{timestamp}_final.mp4"
                mux_audio_video(silent_video, wav_path, final_video)
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
                return final_video

        return silent_video