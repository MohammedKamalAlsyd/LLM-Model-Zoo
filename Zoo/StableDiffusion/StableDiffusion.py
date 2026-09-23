"""Unified Stable Diffusion v1.5 Model Container unifying VAE, UNet, and CLIP text encoders."""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from Zoo.StableDiffusion.configs import StableDiffusionConfig
from Zoo.StableDiffusion.modules.VAE import VAE
from Zoo.StableDiffusion.modules.UNet import UNetModel
from Zoo.CLIP.modules.CLIPTextModel import CLIPTextTransformer
from Zoo.Common.model_loader import auto_detect_device_and_dtype


class StableDiffusionModel(nn.Module):
    """Complete Stable Diffusion v1.5 model container."""

    def __init__(self, config: Optional[StableDiffusionConfig] = None) -> None:
        super().__init__()
        self.config = config or StableDiffusionConfig()

        # 1. Text Tower: Reused directly from Zoo.CLIP
        self.text_encoder = CLIPTextTransformer(self.config.text_config)

        # 2. First-Stage Latent Autoencoder
        self.vae = VAE(self.config.vae_config)

        # 3. Latent Noise Prediction UNet
        self.unet = UNetModel(self.config.unet_config)

    @classmethod
    def from_pretrained_weights(
        cls,
        repo_id: str = "runwayml/stable-diffusion-v1-5",
        filename: str = "v1-5-pruned-emaonly.ckpt",
        config: Optional[StableDiffusionConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "StableDiffusionModel":
        """Downloads standard checkpoint and maps keys cleanly into the unified architecture.

        Args:
            repo_id: Hugging Face repo identifier.
            filename: Target checkpoint file name.
            config: Optional StableDiffusionConfig instance.
            device: Target execution device (auto-detected if None).
            dtype: Target precision dtype (auto-detected if None).

        Returns:
            Fully initialized, weight-populated StableDiffusionModel in eval mode.
        """
        resolved_device, resolved_dtype = auto_detect_device_and_dtype(device, dtype)
        model = cls(config=config).to(dtype=resolved_dtype)

        print(f"Fetching '{filename}' from '{repo_id}'...")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

        print(f"Loading checkpoint weights into RAM (map_location='cpu')...")
        raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = raw_state["state_dict"] if "state_dict" in raw_state else raw_state

        unet_dict: Dict[str, torch.Tensor] = {}
        vae_dict: Dict[str, torch.Tensor] = {}
        clip_dict: Dict[str, torch.Tensor] = {}

        print("Routing standard SD checkpoint keys to unified sub-models...")
        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."):
                unet_dict[key.replace("model.diffusion_model.", "")] = value.to(dtype=resolved_dtype)
            elif key.startswith("first_stage_model."):
                vae_dict[key.replace("first_stage_model.", "")] = value.to(dtype=resolved_dtype)
            elif key.startswith("cond_stage_model.transformer.text_model."):
                clip_dict[key.replace("cond_stage_model.transformer.text_model.", "")] = value.to(dtype=resolved_dtype)

        # Drop unused fixed buffers if present
        clip_dict.pop("embeddings.position_ids", None)

        print("Enforcing strict weight integrity...")
        model.unet.load_state_dict(unet_dict, strict=True)
        model.vae.load_state_dict(vae_dict, strict=True)
        model.text_encoder.load_state_dict(clip_dict, strict=True)

        model.to(device=resolved_device)
        model.eval()
        print(f"✓ StableDiffusionModel successfully loaded on {resolved_device.upper()} in {resolved_dtype}.")
        return model

    def encode_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encodes text token IDs into unpooled contextual embeddings for the UNet.

        Args:
            input_ids: LongTensor of shape (batch, seq_len).

        Returns:
            Context tensor of shape (batch, seq_len, context_dim) (e.g. (B, 77, 768)).
        """
        return self.text_encoder(input_ids, return_pooled=False)