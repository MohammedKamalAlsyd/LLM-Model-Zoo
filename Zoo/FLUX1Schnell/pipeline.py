from typing import Any, List, Optional, Union
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

# Safe fallback for T5TokenizerFast
try:
    from transformers import T5TokenizerFast # type: ignore
except ImportError:
    T5TokenizerFast = AutoTokenizer  

from .SubModels.AutoEncoderKL import AutoencoderKL, DecoderOutput
from .SubModels.FluxTransformer2DModel import FluxTransformer2DModel
from .SubModels.SchedulingFlowMatchEulerDiscrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from .SubModels.T5EncoderModel import T5EncoderModel


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculates the sequence-length-dependent time shift mu for Flow Matching."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Packs 2D spatial latents into 2x2 patch sequences:
    [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
    """
    Unpacks sequence tokens back into 2D spatial latents:
    [B, (H/2)*(W/2), C*4] -> [B, C, H, W]
    """
    b, num_patches, channels = latents.shape
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(b, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(b, channels // 4, h, w)
    return latents


def prepare_latent_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Creates 2D spatial positional coordinates for RoPE: [H/2 * W/2, 3]."""
    h_patches = height // 16
    w_patches = width // 16
    latent_image_ids = torch.zeros(h_patches, w_patches, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] += torch.arange(h_patches, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] += torch.arange(w_patches, device=device, dtype=dtype)[None, :]
    return latent_image_ids.reshape(h_patches * w_patches, 3)


class FluxPipeline:
    """
    Minimal, standalone FLUX.1 [schnell] text-to-image pipeline.
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: nn.Module,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: Any,
        transformer: FluxTransformer2DModel,
    ):
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.transformer = transformer

        self.vae_scale_factor = 8
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None):
        self.device = torch.device(device)
        self.dtype = dtype or torch.bfloat16

        self.vae.to(self.device, self.dtype)
        self.text_encoder.to(self.device, self.dtype)
        self.text_encoder_2.to(self.device, self.dtype)
        self.transformer.to(self.device, self.dtype)
        return self

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts pooled embeddings from CLIP-L and token embeddings from T5-XXL.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # 1. CLIP pooled embedding (pooled_projections)
        clip_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs["input_ids"].to(self.device)
        clip_output = self.text_encoder(input_ids=clip_input_ids)
        pooled_prompt_embeds = getattr(clip_output, "pooler_output", clip_output[1]).to(
            dtype=self.dtype, device=self.device
        )

        # 2. T5 token sequence embeddings
        t5_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs["input_ids"].to(self.device)
        t5_output = self.text_encoder_2(input_ids=t5_input_ids)
        prompt_embeds = t5_output.last_hidden_state.to(dtype=self.dtype, device=self.device)

        # 3. Text RoPE IDs (all zeros for 1D text)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        return prompt_embeds, pooled_prompt_embeds, txt_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,  # Default for schnell
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 256,
    ) -> List[Image.Image]:
        # 1. Validate dimensions (must be divisible by 16)
        height = 2 * (int(height) // 16) * 8
        width = 2 * (int(width) // 16) * 8
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # 2. Encode prompt
        prompt_embeds, pooled_prompt_embeds, txt_ids = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )

        # 3. Sample initial Gaussian noise latents
        latent_channels = 16
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        noise = torch.randn(
            (batch_size, latent_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = pack_latents(noise)
        img_ids = prepare_latent_image_ids(height, width, device=self.device, dtype=self.dtype)

        # 4. Configure Flow-Match Schedule
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len)
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=self.device, mu=mu)
        self.scheduler.set_begin_index(0)

        # 5. Denoising loop
        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Velocity prediction (FLUX.1 Schnell uses guidance_scale = 0.0, so guidance is None)
            model_output = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=timestep / 1000.0,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                return_dict=True,
            ).sample

            # Euler integration step: x_{t - dt} = x_t + dt * v
            step_output = self.scheduler.step(model_output, t, latents, return_dict=True)
            if isinstance(step_output, FlowMatchEulerDiscreteSchedulerOutput):
                latents = step_output.prev_sample
            else:
                latents = step_output[0]

        # 6. Unpack latents and decode with VAE
        latents = unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = self.vae.unscale_latents(latents)

        dec_output = self.vae.decode(latents, return_dict=True)
        sample = dec_output.sample if isinstance(dec_output, DecoderOutput) else dec_output

        # 7. Convert tensor [-1, 1] to PIL Images
        images = (sample / 2.0 + 0.5).clamp(0.0, 1.0)
        images_np = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images_uint8 = (images_np * 255.0).round().astype("uint8")

        return [Image.fromarray(img) for img in images_uint8]