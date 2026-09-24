"""Euler Discrete Scheduler for Flow-Matching architectures with dynamic time shifting."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch

from Zoo.FLUX1Schnell.configs import FluxSchedulerConfig


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput:
    """Output container for the FlowMatch scheduler step."""
    prev_sample: torch.Tensor


class FlowMatchEulerDiscreteScheduler:
    """Discrete Flow-Matching scheduler propagating states along predicted velocity fields."""

    def __init__(self, config: Optional[FluxSchedulerConfig] = None) -> None:
        self.config = config or FluxSchedulerConfig()
        self.num_train_timesteps = self.config.num_train_timesteps
        self.shift = self.config.shift
        self.use_dynamic_shifting = self.config.use_dynamic_shifting
        self.time_shift_type = self.config.time_shift_type

        # Base linspace schedule
        timesteps = np.linspace(1, self.num_train_timesteps, self.num_train_timesteps, dtype=np.float32)[::-1].copy()
        sigmas = timesteps / self.num_train_timesteps
        if not self.use_dynamic_shifting:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        self.sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        self.timesteps = self.sigmas * self.num_train_timesteps

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
        """Configures discrete evaluation timesteps along the Flow Matching trajectory.

        Args:
            num_inference_steps: Number of integration intervals (typically 4 for Schnell).
            device: Target device for allocated scheduling tensors.
            mu: Sequence-dependent dynamic shift coefficient.
            sigmas: Optional explicit sigma schedule.
        """
        self.num_inference_steps = num_inference_steps

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

        if self.use_dynamic_shifting or mu is not None:
            if mu is None:
                raise ValueError("Dynamic shifting requires precalculated 'mu' parameter.")
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        sigmas_tensor = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        terminal_sigma = torch.zeros(1, dtype=torch.float32, device=sigmas_tensor.device)
        self.sigmas = torch.cat([sigmas_tensor, terminal_sigma])

        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps
        self._step_index = None

    def index_for_timestep(self, timestep: Union[float, torch.Tensor]) -> int:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
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
        """Executes a first-order Euler integration step along the predicted velocity field.

        Formula:
            x_{t - dt} = x_t + dt * v_theta
            where dt = sigma_{t+1} - sigma_t (dt < 0)

        Args:
            model_output: Predicted velocity field tensor of shape (B, C, ...).
            timestep: Discrete evaluation timestep index.
            sample: Current noisy sample tensor x_t.
            return_dict: Whether to wrap output in dataclass.

        Returns:
            Denoised sample at next time boundary.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        step_idx = self.step_index
        assert step_idx is not None

        sigma = self.sigmas[step_idx].to(device=sample.device)
        sigma_next = self.sigmas[step_idx + 1].to(device=sample.device)
        dt = sigma_next - sigma

        # Maintain float32 precision during Euler update to eliminate drift
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
        """Forward Flow-Matching corruption: x_t = (1 - sigma) * sample + sigma * noise."""
        step_idx = self.index_for_timestep(timestep)
        sigma = self.sigmas[step_idx].to(device=sample.device, dtype=sample.dtype)
        return sigma * noise + (1.0 - sigma) * sample

    def __len__(self) -> int:
        return self.num_train_timesteps