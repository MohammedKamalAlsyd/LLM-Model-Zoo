# Copyright 2024-2025 The Alibaba Wan Team Authors and Project Contributors.
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

__all__ = [
    "FlowUniPCMultistepScheduler",
    "FlowDPMSolverMultistepScheduler",
    "WanFlowScheduler",
]


def get_sampling_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    return shift * sigma / (1 + (shift - 1) * sigma)


def retrieve_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    Dedicated Multi-step Predictor-Corrector ODE Solver for Flow Matching.
    """

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
    ):
        self.predict_x0 = predict_x0
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None

    @property
    def step_index(self) -> int:
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], np.ndarray]] = None,
        shift: Optional[float] = None,
    ):
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            s = shift if shift is not None else self.config.shift
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device="cpu")
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator=None,
    ) -> Union[SchedulerOutput, Tuple]:
        sigma_t = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        # Flow Matching velocity to x0 data prediction
        x0_pred = sample - sigma_t * model_output

        # UniPC 1st-order / Euler forward ODE update
        dt = sigma_next - sigma_t
        prev_sample = sample + dt * model_output

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)


class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow Matching DPM-Solver++ high-order adaptive ODE stepper.
    """

    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, solver_order: int = 2, shift: float = 1.0, **kwargs):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        self.timesteps = self.sigmas[:-1] * num_train_timesteps
        self._step_index = 0

    def set_timesteps(self, num_inference_steps: int = 50, device=None, sigmas=None, shift=None):
        s = shift or self.shift
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).to(device=device, dtype=torch.int64)
        self._step_index = 0

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor, return_dict: bool = True, **kwargs):
        sigma_t = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = sigma_next - sigma_t
        prev_sample = sample + dt * model_output
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)


# Default scheduler alias for the SubModels namespace
WanFlowScheduler = FlowUniPCMultistepScheduler