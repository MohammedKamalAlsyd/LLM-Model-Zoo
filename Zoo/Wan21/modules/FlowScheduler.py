"""Continuous Flow-Matching ODE Solvers for Wan2.1."""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

__all__ = ["FlowUniPCMultistepScheduler", "FlowDPMSolverMultistepScheduler", "WanFlowScheduler"]


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """Dedicated Multi-step Predictor-Corrector ODE Solver for Flow Matching."""

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        final_sigmas_type: str = "zero",
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.shift = shift
        self.predict_x0 = predict_x0

        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.model_outputs: List[Optional[torch.Tensor]] = [None] * solver_order
        self.timestep_list: List[Optional[torch.Tensor]] = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index: Optional[int] = None

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], np.ndarray]] = None,
        shift: Optional[float] = None,
    ) -> None:
        if sigmas is None:
            assert num_inference_steps is not None
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            s = shift if shift is not None else self.shift
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.sigmas = torch.from_numpy(sigmas).to(device=dev, dtype=torch.float32)
        self.timesteps = torch.from_numpy(timesteps).to(device=dev, dtype=torch.int64)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[SchedulerOutput, Tuple]:
        assert self._step_index is not None
        sigma_t = self.sigmas[self._step_index].to(device=sample.device, dtype=sample.dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=sample.dtype)

        # Flow Matching Euler/UniPC forward update
        dt = sigma_next - sigma_t
        prev_sample = (sample + dt * model_output.to(dtype=sample.dtype)).to(dtype=sample.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)


class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """Flow Matching DPM-Solver++ high-order adaptive ODE stepper."""

    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, solver_order: int = 2, shift: float = 1.0, **kwargs: Any) -> None:
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        self.timesteps = self.sigmas[:-1] * num_train_timesteps
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], np.ndarray]] = None,
        shift: Optional[float] = None,
    ) -> None:
        s = shift or self.shift
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.sigmas = torch.from_numpy(sigmas).to(device=dev, dtype=torch.float32)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).to(device=dev, dtype=torch.int64)
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[SchedulerOutput, Tuple]:
        sigma_t = self.sigmas[self._step_index].to(device=sample.device, dtype=sample.dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=sample.dtype)
        dt = sigma_next - sigma_t
        prev_sample = sample + dt * model_output
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)


WanFlowScheduler = FlowUniPCMultistepScheduler