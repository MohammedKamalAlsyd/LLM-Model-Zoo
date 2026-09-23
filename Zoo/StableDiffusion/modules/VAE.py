"""First-Stage Autoencoder (AutoencoderKL) matching official CompVis/StabilityAI weight keys."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.StableDiffusion.configs import VAEConfig


class Downsample(nn.Module):
    """Spatial downsampling layer applying asymmetric padding before 2D convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Asymmetric padding: (pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling layer combining nearest-neighbor interpolation with 2D convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResnetBlock(nn.Module):
    """Residual convolutional block with pre-activation GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.nin_shortcut(x)

        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))

        return x + res


class AttnBlock(nn.Module):
    """Self-attention block using 1x1 convolutions matching CompVis parameter shapes."""

    def __init__(self, in_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        b, c, h, w = x.shape

        h_norm = self.norm(x)

        # Convert to sequence format for SDPA: (B, 1, H*W, C)
        q = self.q(h_norm).view(b, c, h * w).transpose(1, 2).unsqueeze(1)
        k = self.k(h_norm).view(b, c, h * w).transpose(1, 2).unsqueeze(1)
        v = self.v(h_norm).view(b, c, h * w).transpose(1, 2).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Convert back to spatial image grid: (B, C, H, W)
        out = out.squeeze(1).transpose(1, 2).view(b, c, h, w)
        return res + self.proj_out(out)


class DownBlock(nn.Module):
    """Downsampling stage comprising ResnetBlocks followed by optional strided downsampling."""

    def __init__(self, in_channels: int, out_channels: int, add_downsample: bool) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                ResnetBlock(in_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
            ]
        )
        self.downsample = Downsample(out_channels) if add_downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.block:
            x = resnet(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """Upsampling stage comprising 3 ResnetBlocks followed by optional nearest upsampling."""

    def __init__(self, in_channels: int, out_channels: int, add_upsample: bool) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                ResnetBlock(in_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
            ]
        )
        self.upsample = Upsample(out_channels) if add_upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.block:
            x = resnet(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    """Bottleneck block featuring Resnet-Attention-Resnet topology."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_2(self.attn_1(self.block_1(x)))


class VAE_Encoder(nn.Module):
    """Convolutional image encoder mapping (3, 512, 512) to latent moments (8, 64, 64)."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        base = cfg.base_channels
        self.conv_in = nn.Conv2d(cfg.in_channels, base, kernel_size=3, padding=1)

        self.down = nn.ModuleList(
            [
                DownBlock(base * 1, base * 1, add_downsample=True),   # down.0
                DownBlock(base * 1, base * 2, add_downsample=True),   # down.1
                DownBlock(base * 2, base * 4, add_downsample=True),   # down.2
                DownBlock(base * 4, base * 4, add_downsample=False),  # down.3
            ]
        )
        self.mid = MidBlock(base * 4)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base * 4, eps=cfg.layer_norm_eps)
        self.conv_out = nn.Conv2d(base * 4, cfg.latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down:
            x = block(x)
        x = self.mid(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class VAE_Decoder(nn.Module):
    """Convolutional decoder reconstructing latents (4, 64, 64) into pixel images (3, 512, 512)."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        base = cfg.base_channels
        self.conv_in = nn.Conv2d(cfg.latent_channels, base * 4, kernel_size=3, padding=1)
        self.mid = MidBlock(base * 4)

        self.up = nn.ModuleList(
            [
                UpBlock(base * 2, base * 1, add_upsample=False),  # up.0
                UpBlock(base * 4, base * 2, add_upsample=True),   # up.1
                UpBlock(base * 4, base * 4, add_upsample=True),   # up.2
                UpBlock(base * 4, base * 4, add_upsample=True),   # up.3
            ]
        )
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base * 1, eps=cfg.layer_norm_eps)
        self.conv_out = nn.Conv2d(base * 1, cfg.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid(x)
        for i in reversed(range(4)):
            x = self.up[i](x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class VAE(nn.Module):
    """Complete AutoencoderKL directly."""

    def __init__(self, cfg: Optional[VAEConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = VAE_Encoder(self.cfg)
        self.decoder = VAE_Decoder(self.cfg)
        self.quant_conv = nn.Conv2d(self.cfg.latent_channels * 2, self.cfg.latent_channels * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.cfg.latent_channels, self.cfg.latent_channels, kernel_size=1)

    def encode(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes an RGB image into normalized latent space.

        Args:
            x: Input RGB tensor of shape (batch, 3, height, width) scaled in [-1, 1].
            noise: Gaussian noise tensor of shape (batch, 4, height//8, width//8).

        Returns:
            Latent representation scaled by `scaling_factor` (0.18215).
        """
        moments = self.quant_conv(self.encoder(x))
        mean, log_var = torch.chunk(moments, 2, dim=1)
        log_var = torch.clamp(log_var, min=-30.0, max=20.0)
        std = torch.exp(0.5 * log_var)

        if noise is None:
            noise = torch.randn_like(mean)

        latents = mean + std * noise
        return latents * self.cfg.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes normalized latent vectors back to image space.

        Args:
            z: Latents tensor of shape (batch, 4, height//8, width//8).

        Returns:
            Image tensor of shape (batch, 3, height, width) with values approximately in [-1, 1].
        """
        z = z / self.cfg.scaling_factor
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decode(self.encode(x, noise))