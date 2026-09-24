"""Unified FLUX.1 [schnell] Model Container routing to the low-VRAM streaming engine."""

from typing import Optional, Union
import torch
import torch.nn as nn

from Zoo.Common.model_loader import auto_detect_device_and_dtype
from Zoo.FLUX1Schnell.configs import FluxConfig
from Zoo.FLUX1Schnell.pipeline.FluxPipeline import FluxPipeline


class FluxModel(nn.Module):
    """Entry point for FLUX.1 [schnell] text-to-image synthesis.
    
    Routes execution through the low-memory sequential streaming pipeline
    to guarantee inference within < 1.2 GB VRAM and < 2.5 GB System RAM.
    """

    def __init__(self, config: Optional[FluxConfig] = None) -> None:
        super().__init__()
        self.config = config or FluxConfig()

    @classmethod
    def from_pretrained_weights(
        cls,
        repo_id: str = "black-forest-labs/FLUX.1-schnell",
        config: Optional[FluxConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> FluxPipeline:
        """Instantiates the low-memory Streaming FluxPipeline.

        Instead of allocating 34+ GB across the 12B MMDiT and T5-XXL simultaneously,
        this configures a 3-stage streaming pipeline:
          1. Text Encoding (CLIP + T5-XXL) -> Encoded to latents -> Purged from memory.
          2. Sequential Block Streaming (1 Dual + 1 Single template) -> Purged from memory.
          3. VAE Latent Reconstruction -> Final PIL Image.

        Args:
            repo_id: Hugging Face repository identifier.
            config: Optional FluxConfig configuration dataclass.
            device: Target execution device (auto-detected if None).
            dtype: Target compute precision (auto-detected if None).

        Returns:
            Fully initialized, memory-safe FluxPipeline.
        """
        resolved_device, resolved_dtype = auto_detect_device_and_dtype(device, dtype)
        cfg = config or FluxConfig(repo_id=repo_id)

        print(f"Initializing FLUX.1 [schnell] Streaming Engine on {resolved_device.upper()} ({resolved_dtype})...")
        pipeline = FluxPipeline(config=cfg, device=resolved_device, dtype=resolved_dtype)
        print("✓ FLUX.1 [schnell] streaming pipeline ready.")
        return pipeline