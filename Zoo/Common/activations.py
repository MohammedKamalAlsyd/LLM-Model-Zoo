"""Universal activation functions and registry for LLMs, VLMs, and Vision Backbones."""

from typing import Callable, Dict, Union
import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    """Sigmoid approximation of the Gaussian Error Linear Unit (used in OpenAI CLIP)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ClassInstantier(nn.Module):
    """Helper wrapping functional activations into nn.Module instances."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


# Universal Registry mapping strings to activation factory callables
ACTIVATION_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    # GELU Family
    "quick_gelu": lambda: QuickGELU(),
    "gelu": lambda: nn.GELU(),
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "gelu_accurate": lambda: nn.GELU(approximate="none"),
    # SiLU / Swish Family (LLaMA, Mistral, Qwen, DiT, Flux)
    "silu": lambda: nn.SiLU(),
    "swish": lambda: nn.SiLU(),
    # Standard Non-Linearities
    "relu": lambda: nn.ReLU(),
    "relu6": lambda: nn.ReLU6(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "mish": lambda: nn.Mish(),
    "tanh": lambda: nn.Tanh(),
    "sigmoid": lambda: nn.Sigmoid(),
    # Linear / Identity Pass-through
    "linear": lambda: nn.Identity(),
    "identity": lambda: nn.Identity(),
}


def get_activation(act_fn: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]) -> nn.Module:
    """Universal activation dispatcher resolving string identifiers to PyTorch modules.

    Args:
        act_fn: Identifier string (e.g. 'quick_gelu', 'gelu', 'silu') or an existing module/callable.

    Returns:
        Instantiated nn.Module ready to be plugged into a neural network layer.

    Raises:
        ValueError: If an unknown activation string is provided.
    """
    if isinstance(act_fn, nn.Module):
        return act_fn

    if isinstance(act_fn, str):
        key = act_fn.lower().strip()
        if key in ACTIVATION_REGISTRY:
            return ACTIVATION_REGISTRY[key]()
        raise ValueError(
            f"Unsupported activation '{act_fn}'. Supported: {sorted(list(ACTIVATION_REGISTRY.keys()))}"
        )

    if callable(act_fn):
        return ClassInstantier(act_fn)

    raise TypeError(f"Expected str, nn.Module, or callable, got {type(act_fn)}")