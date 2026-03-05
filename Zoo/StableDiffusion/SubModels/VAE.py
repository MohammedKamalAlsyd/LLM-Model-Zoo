import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================================================
# VAE Sub-Components
# Designed to perfectly match official SD parameter keys natively.
# =================================================================================


class Downsample(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        # Padding is asymmetric in original SD for downsampling: right and bottom by 1
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x) -> torch.Tensor:
        # Pad: (left, right, top, bottom)
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        x = F.interpolate(
            x, scale_factor=2.0, mode="nearest"
        )  # Alternative to UpSample. We Remove it for layer for weights copying purposes.
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection adjustment if channel dimensions change
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x) -> torch.Tensor:
        residual = self.nin_shortcut(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + residual


class AttnBlock(nn.Module):
    """
    Self-Attention block natively using 1x1 convolutions to flawlessly match
    original Stable Diffusion pretrained weights without requiring key mapping/conversion.
    """

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x) -> torch.Tensor:
        residual = x
        B, C, H, W = x.shape

        x = self.norm(x)

        # Convert to (Batch, 1_Head, Seq_Len, Dim) for PyTorch SDPA
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C) -> (B, 1, H*W, C)
        q = self.q(x).view(B, C, H * W).transpose(1, 2).unsqueeze(1)
        k = self.k(x).view(B, C, H * W).transpose(1, 2).unsqueeze(1)
        v = self.v(x).view(B, C, H * W).transpose(1, 2).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Convert back to spatial dimensions: (B, 1, H*W, C) -> (B, C, H, W)
        out = out.squeeze(1).transpose(1, 2).view(B, C, H, W)
        out = self.proj_out(out)

        return out + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, add_downsample) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                ResnetBlock(in_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
            ]
        )
        self.downsample = Downsample(out_channels) if add_downsample else None

    def forward(self, x) -> torch.Tensor:
        for resnet in self.block:
            x = resnet(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, add_upsample) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                ResnetBlock(in_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
                ResnetBlock(
                    out_channels, out_channels
                ),  # UpBlock consistently uses 3 resnets
            ]
        )
        self.upsample = Upsample(out_channels) if add_upsample else None

    def forward(self, x) -> torch.Tensor:
        for resnet in self.block:
            x = resnet(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def forward(self, x) -> torch.Tensor:
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x


# =================================================================================
# Main VAE Models (Encoder & Decoder)
# =================================================================================


class VAE_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        # ModuleList creates keys corresponding strictly to down.0, down.1, down.2, down.3
        self.down = nn.ModuleList(
            [
                DownBlock(128, 128, add_downsample=True),  # down.0
                DownBlock(128, 256, add_downsample=True),  # down.1
                DownBlock(256, 512, add_downsample=True),  # down.2
                DownBlock(512, 512, add_downsample=False),  # down.3
            ]
        )

        self.mid = MidBlock(512)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6)
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)

        for block in self.down:
            x = block(x)

        x = self.mid(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class VAE_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)

        self.mid = MidBlock(512)

        # Keys exactly correspond to up.0, up.1, up.2, up.3
        self.up = nn.ModuleList(
            [
                UpBlock(256, 128, add_upsample=False),  # up.0
                UpBlock(512, 256, add_upsample=True),  # up.1
                UpBlock(512, 512, add_upsample=True),  # up.2
                UpBlock(512, 512, add_upsample=True),  # up.3
            ]
        )

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)

        x = self.mid(x)

        # Decoder executes "up" blocks in reverse (from up.3 down to up.0)
        for i in reversed(range(4)):
            x = self.up[i](x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


# =================================================================================
# Full Top-Level Wrapper
# =================================================================================


class VAE(nn.Module):
    """
    Main VAE wrapper representing `first_stage_model` from the SD Checkpoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1, padding=0)

    def encode(self, x, noise) -> torch.Tensor:
        """
        Takes raw image tensors -> encodes to latents -> applies reparameterization
        Returns latents scaled to match SD pipeline requirements.
        """
        x = self.encoder(x)
        x = self.quant_conv(x)

        # Split into mean and log_variance
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, min=-30.0, max=20.0)
        std = torch.exp(log_var / 2)

        # Reparameterization trick
        x = mean + std * noise

        # Scaling constant standardized in Stable Diffusion pipelines
        x *= 0.18215

        return x

    def decode(self, x) -> torch.Tensor:
        """
        Takes scaled latents -> un-scales -> decodes back to images.
        """
        # Undo SD scaling factor
        x = x / 0.18215

        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x

    def forward(self, x, noise):
        latent = self.encode(x, noise)
        return self.decode(latent)
