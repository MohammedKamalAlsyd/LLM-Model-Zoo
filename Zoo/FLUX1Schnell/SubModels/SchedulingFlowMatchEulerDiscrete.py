import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput:
    """Output class for FlowMatchEulerDiscreteScheduler step function."""
    prev_sample: torch.Tensor


class FlowMatchEulerDiscreteScheduler:
    """
    Euler Discrete Scheduler for Flow-Matching models (e.g., FLUX.1 [schnell]).
    Propagates the sample along the predicted velocity vector field.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        time_shift_type: str = "exponential",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.time_shift_type = time_shift_type

        # Default initial linspace schedule
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        self.timesteps = self.sigmas * num_train_timesteps

        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def _time_shift_exponential(self, mu: float, sigma_power: float, t: np.ndarray) -> np.ndarray:
        exp_mu = math.exp(mu)
        return exp_mu / (exp_mu + (1.0 / t - 1.0) ** sigma_power)

    def _time_shift_linear(self, mu: float, sigma_power: float, t: np.ndarray) -> np.ndarray:
        return mu / (mu + (1.0 / t - 1.0) ** sigma_power)

    def time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        if self.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)
        raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        mu: Optional[float] = None,
        sigmas: Optional[np.ndarray] = None,
    ) -> None:
        """
        Sets the discrete timesteps for the generation trajectory.
        For FLUX [schnell], num_inference_steps is typically 4.
        """
        self.num_inference_steps = num_inference_steps

        # 1. Base linear sigmas from 1.0 down to 1/num_train_timesteps
        if sigmas is None:
            timesteps = np.linspace(
                self.num_train_timesteps,
                self.num_train_timesteps / num_inference_steps,
                num_inference_steps,
                dtype=np.float32,
            )
            sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = np.array(sigmas, dtype=np.float32)

        # 2. Dynamic or static shifting
        if self.use_dynamic_shifting or mu is not None:
            if mu is None:
                raise ValueError("`mu` must be provided when dynamic shifting is enabled.")
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        # 3. Convert to tensor and append terminal sigma (0.0)
        sigmas_tensor = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        terminal_sigma = torch.zeros(1, dtype=torch.float32, device=sigmas_tensor.device)
        self.sigmas = torch.cat([sigmas_tensor, terminal_sigma])

        # Timesteps associated with each sigma
        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps
        self._step_index = None

    def index_for_timestep(self, timestep: Union[float, torch.Tensor]) -> int:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
        # Find the closest matching index in timesteps
        dists = torch.abs(self.timesteps - timestep)
        return int(torch.argmin(dists).item())

    def _init_step_index(self, timestep: Union[float, torch.Tensor]) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Euler forward step along the predicted velocity vector field:
        x_{t - dt} = x_t + dt * v_theta
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        step_idx = self.step_index
        assert step_idx is not None

        sigma = self.sigmas[step_idx].to(device=sample.device)
        sigma_next = self.sigmas[step_idx + 1].to(device=sample.device)

        # dt is negative because we are stepping from noise (sigma=1) to clean image (sigma=0)
        dt = sigma_next - sigma

        # Upcast to float32 to prevent numerical instability during Euler step
        prev_sample = sample.to(torch.float32) + dt * model_output.to(torch.float32)
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index = step_idx + 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow-matching forward process:
        x_t = sigma * noise + (1 - sigma) * sample
        """
        step_idx = self.index_for_timestep(timestep)
        sigma = self.sigmas[step_idx].to(device=sample.device, dtype=sample.dtype)
        return sigma * noise + (1.0 - sigma) * sample

    def __len__(self) -> int:
        return self.num_train_timesteps