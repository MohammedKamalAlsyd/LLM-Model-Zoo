import torch
import numpy as np


# See this First: https://www.youtube.com/watch?v=EhndHhIvWWw
class DDPMSampler:
    """
    Denoising Diffusion Probabilistic Models (DDPM) Sampler.
    Handles the mathematical forward process (adding noise) and the reverse process (denoising).
    """

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.generator = generator
        self.num_train_timesteps = num_training_steps

        # 1. Define the noise schedule (betas).
        # Scaled linearly from beta_start to beta_end.
        # Params match: https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )

        # 2. Calculate alphas and cumulative alphas.
        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # α_bar_t = Π(α_i) from i=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Base timesteps for training: [999, 998, ..., 0]
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        """
        Scales the 1000 training timesteps down to the desired number of inference steps.
        E.g., for 50 steps, it jumps by 20: [980, 960, 940, ..., 0]
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps

        # Calculate the exact timesteps to evaluate during generation
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def set_strength(self, strength: float = 1.0):
        """
        Used for Image-to-Image. Sets how much noise to add to the input image.
        strength = 1.0 : Full noise (Start from scratch, ignore input image)
        strength = 0.0 : No noise (Output exactly matches input image)
        """
        # Calculate how many steps to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)

        # Truncate the timesteps list so we start denoising from the middle
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def add_noise(
        self, original_samples: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Diffusion Process: Jumps directly to timestep 't' and adds the exact amount of noise.
        See Equation (4) from the DDPM paper: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1 - alpha_bar_t)*I)
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timestep = timestep.to(original_samples.device)

        # Get sqrt(α_bar_t) -> Used to scale the original image
        sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5

        # Get sqrt(1 - α_bar_t) -> Used to scale the noise
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5

        # Broadcast shapes to match the latents: (Batch_Size, 4, Height, Width)
        # Using .view(-1, 1, 1, 1) is much cleaner and faster than while loops with unsqueeze
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)

        # Generate standard Gaussian noise: ε ~ N(0, 1)
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        # x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
        noisy_samples = (sqrt_alpha_prod * original_samples) + (
            sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples

    def step(
        self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse Diffusion Process: Takes a noisy latent x_t, and predicts the slightly less noisy latent x_{t-1}.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Fetch alphas and betas for current step
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Compute the predicted original sample (x_0) from the predicted noise.
        # Equation (15): x_0 = (x_t - sqrt(1 - α_bar_t) * ε_theta) / sqrt(α_bar_t)
        pred_original_sample = (latents - (beta_prod_t**0.5) * model_output) / (
            alpha_prod_t**0.5
        )

        # 3. Compute coefficients for x_0 and x_t
        # Equation (7): Formula for the mean mu_tilde_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = (current_alpha_t**0.5 * beta_prod_t_prev) / beta_prod_t

        # 4. Compute predicted previous sample mean (µ_t)
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        # 5. Add Variance (Noise) for the Langevin dynamics
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )

            # Compute variance as per Equation (7)
            variance = (self._get_variance(t) ** 0.5) * noise

        # x_{t-1} = µ_t + σ_t * z
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Calculates the timestep exactly one inference step prior.
        """
        prev_t = timestep - (self.num_train_timesteps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Calculates the variance (σ_t^2) for the reverse process.
        See Equation (7) from the DDPM paper.
        """
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        )
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Compute variance: β_tilde_t = (1 - α_bar_{t-1}) / (1 - α_bar_t) * β_t
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Clamp variance to prevent log(0) issues if variance goes to exactly 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
