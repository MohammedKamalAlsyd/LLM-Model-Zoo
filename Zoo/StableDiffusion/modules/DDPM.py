"""Denoising Diffusion Probabilistic Models (DDPM) Sampler preserving exact mathematical schedules."""

from typing import Optional, Union
import numpy as np
import torch

from Zoo.StableDiffusion.configs import DDPMConfig


class DDPMSampler:
    """Discrete-time Gaussian diffusion scheduler for training and ancestral inference."""

    def __init__(
        self,
        generator: Optional[torch.Generator] = None,
        config: Optional[DDPMConfig] = None,
    ) -> None:
        """Initializes noise schedule, cumulative variance products, and timesteps.

        Args:
            generator: Optional PyTorch pseudorandom number generator for reproducibility.
            config: DDPMConfig containing timesteps and beta boundaries.
        """
        self.config = config or DDPMConfig()
        self.generator = generator
        self.num_train_timesteps = self.config.num_train_timesteps

        # 1. Quadratic interpolation for scaled linear schedule: beta ~ t
        if self.config.beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(
                    self.config.beta_start**0.5,
                    self.config.beta_end**0.5,
                    self.num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        else:
            self.betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.num_train_timesteps,
                dtype=torch.float32,
            )

        # 2. Precompute alpha schedules
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 3. Base training timesteps: [999, 998, ..., 0]
        self.timesteps = torch.from_numpy(np.arange(0, self.num_train_timesteps)[::-1].copy())
        self.num_inference_steps: int = self.num_train_timesteps
        self.start_step: int = 0

    def set_inference_timesteps(self, num_inference_steps: int = 50) -> None:
        """Sub-samples the training schedule down to discrete evaluation timesteps.

        Args:
            num_inference_steps: Number of denoising iterations (e.g., 20, 50).
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps

        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def set_strength(self, strength: float = 1.0) -> None:
        """Truncates timesteps according to image-to-image noise corruption strength.

        Args:
            strength: Multiplier in (0, 1.0]. 1.0 means pure noise prefill.
        """
        if not (0.0 < strength <= 1.0):
            raise ValueError(f"Strength must be in (0.0, 1.0], received {strength}")

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def add_noise(
        self,
        original_samples: torch.Tensor,
        timestep: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward diffusion process: Corrupts clean sample x_0 into x_t analytically.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            original_samples: Clean latent tensor of shape (batch, channels, height, width).
            timestep: Discrete integer or scalar tensor specifying noise magnitude.

        Returns:
            Noisy latent tensor matching input shape.
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if isinstance(timestep, int):
            t_idx = timestep
        else:
            t_idx = int(timestep.item())

        sqrt_alpha_prod = (alphas_cumprod[t_idx] ** 0.5).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = ((1.0 - alphas_cumprod[t_idx]) ** 0.5).view(-1, 1, 1, 1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        return (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod * noise)

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse diffusion ancestral transition: Computes x_{t-1} given x_t and eps_theta.

        Args:
            timestep: Current diffusion timestep integer t.
            latents: Current latent state x_t of shape (batch, channels, height, width).
            model_output: Predicted noise residual eps_theta of matching shape.

        Returns:
            Less noisy latent state x_{t-1}.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Fetch variance schedules for t and prev_t
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)

        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # 2. Estimate original sample x_0 via Tweedie's formula:
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_original_sample = (latents - (beta_prod_t**0.5) * model_output) / (alpha_prod_t**0.5)

        # 3. Compute posterior mean mu_tilde_t(x_t, x_0)
        pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t**0.5 * beta_prod_t_prev) / beta_prod_t

        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        # 4. Langevin stochastic variance injection
        variance = 0.0
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance

    def _get_previous_timestep(self, timestep: int) -> int:
        """Finds the previous timestep along the discrete sub-sampled trajectory."""
        return timestep - (self.num_train_timesteps // self.num_inference_steps)

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """Analytically computes posterior variance beta_tilde_t."""
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev

        # Posterior variance: beta_tilde_t = ((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * beta_t
        variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t
        return torch.clamp(variance, min=1e-20)