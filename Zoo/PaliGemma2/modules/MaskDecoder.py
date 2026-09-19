"""UViM VQ-VAE Mask Decoder for PaliGemma 2 segmentation tokens."""

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class ResBlock(nn.Module):
    """Residual block matching Google's UViM 3-conv design."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PaliGemmaMaskDecoder(nn.Module):
    """Reconstructs 64x64 binary instance masks from 16 discrete VQ-VAE tokens."""

    def __init__(self, codebook_size: int = 128, embed_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)

        # Exact PyTorch layer hierarchy matching keys in vae-oid.npz
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=1, padding=0),                # 0
            nn.ReLU(inplace=True),                                              # 1
            ResBlock(128),                                                      # 2 (decoder.2.net)
            ResBlock(128),                                                      # 3 (decoder.3.net)
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 4: 4x4 -> 8x8
            nn.ReLU(inplace=True),                                              # 5
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 6: 8x8 -> 16x16
            nn.ReLU(inplace=True),                                              # 7
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 8: 16x16 -> 32x32
            nn.ReLU(inplace=True),                                              # 9
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 10: 32x32 -> 64x64
            nn.ReLU(inplace=True),                                              # 11
            nn.Conv2d(16, 1, kernel_size=1, padding=0),                         # 12: 1-channel mask output
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:

            token_ids: Tensor of shape (B, 16) with integer IDs in [0, 127].

        Returns:
            Tensor of shape (B, 64, 64) with raw mask logits in [-1, 1].
        """
        b = token_ids.shape[0]
        # (B, 16) -> (B, 16, 512) -> (B, 4, 4, 512) -> (B, 512, 4, 4)
        x = self.embedding(token_ids).view(b, 4, 4, -1).permute(0, 3, 1, 2)
        logits = self.decoder(x).squeeze(1)  # (B, 64, 64)
        return logits


def load_mask_decoder(device: str = "cpu", dtype: torch.dtype = torch.float32) -> PaliGemmaMaskDecoder:
    """Downloads and loads official Google UViM VQ-VAE weights."""
    model = PaliGemmaMaskDecoder().to(device=device, dtype=dtype)

    print("Fetching UViM Mask VAE weights (vae-oid.npz) from Hugging Face...")
    weights_path = hf_hub_download(
        repo_id="big-vision/paligemma-hf",
        filename="vae-oid.npz",
        repo_type="space",
    )

    npz = np.load(weights_path)
    state_dict = model.state_dict()

    # Load 512-dim embedding codebook
    if "_vq_vae._embedding" in npz:
        state_dict["embedding.weight"].copy_(
            torch.from_numpy(npz["_vq_vae._embedding"]).to(device=device, dtype=dtype)
        )

    # The checkpoint keys (decoder.0, decoder.2.net, decoder.4, etc.)
    # map 1:1 to self.decoder
    for key in state_dict.keys():
        if key in npz:
            state_dict[key].copy_(
                torch.from_numpy(npz[key]).to(device=device, dtype=dtype)
            )

    model.eval()
    print("✓ UViM Mask VAE Decoder successfully initialized.")
    return model