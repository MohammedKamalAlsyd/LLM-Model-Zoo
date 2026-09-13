import gc
from typing import Optional, cast
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .SubModels.AutoEncoderKL import AutoencoderKL, DecoderOutput
from .SubModels.FluxTransformer2DModel import FluxTransformer2DModel, Transformer2DModelOutput
from .SubModels.SchedulingFlowMatchEulerDiscrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from .SubModels.T5EncoderModel import T5EncoderModel
from .pipeline import calculate_shift, pack_latents, prepare_latent_image_ids, unpack_latents
from .utils.model_loader import get_safetensors_files, stream_safetensors_to_model


def purge_memory():
    """Forces aggressive garbage collection and releases cached CUDA blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class StreamingFluxPipeline:
    """
    Memory-efficient FLUX.1 [schnell] runner designed for systems with <= 30GB total memory.
    Streams weights into GPU directly from disk and purges models between stages.
    """

    def __init__(self, checkpoint_path: str = "black-forest-labs/FLUX.1-schnell", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.vae_scale_factor = 8

        print("Locating repository shards on Hugging Face Hub / local cache...")
        self.transformer_files = get_safetensors_files(checkpoint_path, subfolder="transformer")
        self.t5_files = get_safetensors_files(checkpoint_path, subfolder="text_encoder_2")
        self.vae_files = get_safetensors_files(checkpoint_path, subfolder="vae")

        # Scheduler is lightweight (<1 MB) and safe to keep permanently in memory
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            time_shift_type="linear",
        )

    # --------------------------------------------------------------------------
    # STAGE 1: Text Encoding (Peaks at ~10 GB VRAM, < 1.5 GB RAM)
    # --------------------------------------------------------------------------
    def _encode_prompt(
        self, prompt: str, max_sequence_length: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("\n[Stage 1/3] Loading CLIP-L and T5-XXL for Text Encoding...")

        # 1. CLIP-L Pooled Encoding (~0.5 GB)
        tokenizer = CLIPTokenizer.from_pretrained(self.checkpoint_path, subfolder="tokenizer")
        
        # Explicit cast to nn.Module resolves Pylance unbound method / __call__ inference issue
        raw_clip = CLIPTextModel.from_pretrained(
            self.checkpoint_path, subfolder="text_encoder", torch_dtype=self.dtype
        )
        text_encoder = cast(nn.Module, raw_clip).to(self.device)

        clip_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_output = text_encoder(input_ids=clip_inputs["input_ids"].to(self.device))
        pooled_prompt_embeds = getattr(clip_output, "pooler_output", clip_output[1]).to(
            dtype=self.dtype, device=self.device
        )

        del text_encoder, tokenizer, raw_clip
        purge_memory()

        # 2. T5-XXL Token Encoding (~9.5 GB)
        tokenizer_2 = AutoTokenizer.from_pretrained(self.checkpoint_path, subfolder="tokenizer_2")
        with torch.device("meta"):
            t5_encoder = T5EncoderModel()

        print("  Streaming T5-XXL directly to GPU...")
        stream_safetensors_to_model(t5_encoder, self.t5_files, device=self.device, dtype=self.dtype)

        t5_inputs = tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_output = t5_encoder(input_ids=t5_inputs["input_ids"].to(self.device))
        prompt_embeds = t5_output.last_hidden_state.to(dtype=self.dtype, device=self.device)

        # 3. Text RoPE IDs
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        # Immediate cleanup of Stage 1 models
        del t5_encoder, tokenizer_2, t5_output
        purge_memory()
        print("  Prompt encoded successfully. All text encoders purged from memory.")
        return prompt_embeds, pooled_prompt_embeds, txt_ids

    # --------------------------------------------------------------------------
    # STAGE 2: Transformer Denoising (Peaks at ~24.5 GB VRAM, < 1.5 GB RAM)
    # --------------------------------------------------------------------------
    def _denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        num_inference_steps: int = 4,
    ) -> torch.Tensor:
        print("\n[Stage 2/3] Constructing and streaming 12B Transformer to GPU (~24 GB)...")

        with torch.device("meta"):
            transformer = FluxTransformer2DModel(
                patch_size=1,
                in_channels=64,
                num_layers=19,
                num_single_layers=38,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=4096,
                pooled_projection_dim=768,
                guidance_embeds=False,
            )

        stream_safetensors_to_model(transformer, self.transformer_files, device=self.device, dtype=self.dtype)
        print("  Transformer weights streamed into GPU. Starting Flow-Matching Euler steps...")

        # Setup Euler Schedule
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len)
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=self.device, mu=mu)
        self.scheduler.set_begin_index(0)

        for step_idx, t in enumerate(self.scheduler.timesteps):
            print(f"  Denoising step {step_idx + 1}/{num_inference_steps} (timestep {t.item():.1f})...")
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with torch.no_grad():
                transformer_res = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timestep / 1000.0,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=None,
                    return_dict=True,
                )
                
                model_output = (
                    transformer_res.sample
                    if isinstance(transformer_res, Transformer2DModelOutput)
                    else transformer_res[0]
                )

                step_output = self.scheduler.step(model_output, t, latents, return_dict=True)
                if isinstance(step_output, FlowMatchEulerDiscreteSchedulerOutput):
                    latents = step_output.prev_sample
                else:
                    latents = step_output[0]

        # Immediate cleanup of Stage 2 model
        del transformer
        purge_memory()
        print("  Denoising complete. Transformer completely purged from GPU memory.")
        return latents

    # --------------------------------------------------------------------------
    # STAGE 3: VAE Decoding (Peaks at ~1.0 GB VRAM, < 1.5 GB RAM)
    # --------------------------------------------------------------------------
    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> Image.Image:
        print("\n[Stage 3/3] Streaming VAE to GPU and decoding final image...")

        with torch.device("meta"):
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=16,
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                scaling_factor=0.3611,
                shift_factor=0.1159,
            )

        stream_safetensors_to_model(vae, self.vae_files, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            latents = unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = vae.unscale_latents(latents)
            dec_output = vae.decode(latents, return_dict=True)
            sample = dec_output.sample if isinstance(dec_output, DecoderOutput) else dec_output

            images = (sample / 2.0 + 0.5).clamp(0.0, 1.0)
            images_np = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images_uint8 = (images_np * 255.0).round().astype("uint8")

        del vae
        purge_memory()
        print("  Decoding finished. VAE purged from memory.")
        return Image.fromarray(images_uint8[0])

    # --------------------------------------------------------------------------
    # Pipeline Call
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        height = 2 * (int(height) // 16) * 8
        width = 2 * (int(width) // 16) * 8

        # Stage 1: Encode Text
        prompt_embeds, pooled_prompt_embeds, txt_ids = self._encode_prompt(prompt)

        # Prepare Latent Noise
        latent_channels = 16
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        noise = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = pack_latents(noise)
        img_ids = prepare_latent_image_ids(height, width, device=self.device, dtype=self.dtype)

        # Stage 2: Denoise with Transformer
        latents = self._denoise_latents(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            num_inference_steps=num_inference_steps,
        )

        # Stage 3: Decode with VAE
        image = self._decode_latents(latents, height, width)
        return image


# ==============================================================================
# Execution Entry Point
# ==============================================================================
if __name__ == "__main__":
    prompt = (
        "A sleek cybernetic robotic tiger walking through a rain-slicked Tokyo street at night, "
        "neon reflections, 8k resolution, cinematic lighting"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemError("CUDA GPU is required to run FLUX under the specified memory budget.")

    pipeline = StreamingFluxPipeline(device=device)

    generator = torch.Generator(device=device).manual_seed(42)
    output_image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        generator=generator,
    )

    output_filename = "flux_output.png"
    output_image.save(output_filename)
    print(f"\nCompleted! Generated image saved to {output_filename}")