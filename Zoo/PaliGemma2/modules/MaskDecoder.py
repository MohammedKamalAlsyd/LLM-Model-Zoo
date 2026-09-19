"""UViM VQ-VAE Mask Decoder for PaliGemma 2 segmentation tokens."""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class PaliGemmaMaskDecoder(nn.Module):
    """Reconstructs 64x64 binary instance masks from 16 discrete VQ-VAE tokens."""

    def __init__(self, codebook_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)

        # 4x4 (dim 128)
        self.conv_in = nn.Conv2d(embed_dim, 128, kernel_size=1, padding=0)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)

        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  # -> 8x8
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # -> 16x16
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # -> 32x32
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)    # -> 64x64

        self.conv_out = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:

            token_ids: Tensor of shape (B, 16) with integer IDs in [0, 127].

        Returns:
            Tensor of shape (B, 64, 64) with mask probabilities in [0.0, 1.0].
        """
        b = token_ids.shape[0]
        # (B, 16) -> (B, 16, 64) -> (B, 4, 4, 64) -> (B, 64, 4, 4)
        x = self.embedding(token_ids).view(b, 4, 4, -1).permute(0, 3, 1, 2)

        x = F.relu(self.conv_in(x))
        x = self.res1(x)
        x = self.res2(x)

        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))

        logits = self.conv_out(x).squeeze(1)  # (B, 64, 64)
        return torch.sigmoid(logits)


def load_mask_decoder(device: str = "cpu", dtype: torch.dtype = torch.float32) -> PaliGemmaMaskDecoder:
    """Downloads and loads official Google UViM VQ-VAE weights from Hugging Face."""
    model = PaliGemmaMaskDecoder().to(device=device, dtype=dtype)

    print("Fetching UViM Mask VAE weights (vae-oid.npz) from Hugging Face...")
    weights_path = hf_hub_download(
        repo_id="big-vision/paligemma-hf",
        filename="vae-oid.npz",
        repo_type="space",
    )

    # Load parameters from numpy archive
    npz = np.load(weights_path)
    state_dict = model.state_dict()

    # Load codebook embeddings
    if "_vq_vae._embedding" in npz:
        state_dict["embedding.weight"].copy_(
            torch.from_numpy(npz["_vq_vae._embedding"]).to(device=device, dtype=dtype)
        )

    # Load Conv & ConvTranspose layers (weights stored in PyTorch standard layout)
    key_mapping = {
        "decoder.0.weight": "conv_in.weight",
        "decoder.0.bias": "conv_in.bias",
        "decoder.2.net.0.weight": "res1.conv1.weight",
        "decoder.2.net.0.bias": "res1.conv1.bias",
        "decoder.2.net.2.weight": "res1.conv2.weight",
        "decoder.2.net.2.bias": "res1.conv2.bias",
        "decoder.3.net.0.weight": "res2.conv1.weight",
        "decoder.3.net.0.bias": "res2.conv1.bias",
        "decoder.3.net.2.weight": "res2.conv2.weight",
        "decoder.3.net.2.bias": "res2.conv2.bias",
        "decoder.4.weight": "up1.weight",
        "decoder.4.bias": "up1.bias",
        "decoder.6.weight": "up2.weight",
        "decoder.6.bias": "up2.bias",
        "decoder.8.weight": "up3.weight",
        "decoder.8.bias": "up3.bias",
        "decoder.10.weight": "up4.weight",
        "decoder.10.bias": "up4.bias",
        "decoder.12.weight": "conv_out.weight",
        "decoder.12.bias": "conv_out.bias",
    }

    for np_key, pt_key in key_mapping.items():
        if np_key in npz and pt_key in state_dict:
            arr = npz[np_key]
            state_dict[pt_key].copy_(torch.from_numpy(arr).to(device=device, dtype=dtype))

    model.eval()
    print("✓ UViM Mask VAE Decoder successfully initialized.")
    return model