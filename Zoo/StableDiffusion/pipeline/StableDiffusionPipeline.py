"""Inference pipeline for Stable Diffusion v1.5 text-to-image and image-to-image generation."""

from typing import Optional, Union
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

from Zoo.StableDiffusion.StableDiffusion import StableDiffusionModel
from Zoo.StableDiffusion.modules.DDPM import DDPMSampler


def rescale(
    x: torch.Tensor,
    old_range: tuple,
    new_range: tuple,
    clamp: bool = False,
) -> torch.Tensor:
    """Linearly maps a tensor between dynamic numerical intervals."""
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = (x - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generates continuous sinusoidal frequency embeddings for a discrete diffusion timestep.

    Args:
        timestep: Integer timestep index in [0, 999].
        device: Device to place the returned tensor on.
        dtype: Target floating point precision.

    Returns:
        Tensor of shape (1, 320) combining cos and sin frequencies.
    """
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32, device=device) / 160)
    x = torch.tensor([timestep], dtype=torch.float32, device=device)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1).to(dtype=dtype)


class StableDiffusionPipeline:
    """Production inference pipeline managing latent trajectories and multimodal condition binding."""

    def __init__(
        self,
        model: StableDiffusionModel,
        tokenizer: Optional[CLIPTokenizer] = None,
        sampler: Optional[DDPMSampler] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.sampler = sampler or DDPMSampler()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        uncond_prompt: str = "",
        input_image: Optional[Image.Image] = None,
        strength: float = 0.8,
        do_cfg: bool = True,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Executes unconditional/conditional ancestral denoising.

        Args:
            prompt: Text description of target image.
            uncond_prompt: Negative conditioning prompt.
            input_image: Optional PIL Image input for Image-to-Image translation.
            strength: Degree of noise corruption for Image-to-Image in (0, 1.0].
            do_cfg: Enable Classifier-Free Guidance.
            cfg_scale: Guidance amplification coefficient (typically 7.0 - 8.5).
            num_inference_steps: Discrete DDPM evaluation intervals.
            height: Image height in pixels (multiple of 8).
            width: Image width in pixels (multiple of 8).
            seed: Optional integer seed for reproducibility.

        Returns:
            Rendered PIL Image in RGB format.
        """
        # 1. Generator and Seed Setup
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator.seed()

        self.sampler.generator = generator
        self.sampler.set_inference_timesteps(num_inference_steps)

        # 2. Text Conditioning via CLIP
        cond_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        cond_context = self.model.encode_prompt(cond_tokens)  # (1, 77, 768)

        if do_cfg:
            uncond_tokens = self.tokenizer(
                uncond_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)

            uncond_context = self.model.encode_prompt(uncond_tokens)  # (1, 77, 768)
            context = torch.cat([cond_context, uncond_context], dim=0)  # (2, 77, 768)
        else:
            context = cond_context

        # 3. Latent Initialization (Img2Img vs Txt2Img)
        latents_shape = (1, 4, height // 8, width // 8)

        if input_image is not None:
            # Aspect resize & normalization to [-1, 1]
            img = input_image.convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(device=self.device, dtype=self.dtype)
            img_tensor = rescale(img_tensor, (0.0, 255.0), (-1.0, 1.0)).unsqueeze(0)

            # Encode image to latent moments
            encoder_noise = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.dtype)
            latents = self.model.vae.encode(img_tensor, encoder_noise)

            # Schedule noise injection
            self.sampler.set_strength(strength)
            latents = self.sampler.add_noise(latents, self.sampler.timesteps[0])
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.dtype)

        # 4. Ancestral Denoising Trajectory
        for timestep in tqdm(self.sampler.timesteps, desc="Sampling Timesteps"):
            t_int = int(timestep.item())
            time_emb = get_time_embedding(t_int, device=self.device, dtype=self.dtype)

            # Classifier-Free Guidance Batch Duplication
            model_input = latents.repeat(2, 1, 1, 1) if do_cfg else latents

            # UNet Noise Residual Prediction
            model_output = self.model.unet(model_input, context, time_emb)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

            # Analytical Sampler Step
            latents = self.sampler.step(t_int, latents, model_output)

        # 5. Latent Space Decoding to RGB
        decoded = self.model.vae.decode(latents)

        # 6. Postprocessing to PIL
        decoded = rescale(decoded, (-1.0, 1.0), (0.0, 255.0), clamp=True)
        img_array = decoded.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()[0]
        return Image.fromarray(img_array)