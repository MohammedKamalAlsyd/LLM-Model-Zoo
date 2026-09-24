"""FLUX AutoencoderKL (VAE) implementation with 16 latent channels and tiled evaluation."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.FLUX1Schnell.configs import FluxVAEConfig


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class DiagonalGaussianDistribution:
    """Diagonal Gaussian parameterization with clamped log-variance."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        eps = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class AutoencoderKLOutput:
    latent_dist: DiagonalGaussianDistribution


class ResnetBlock2D(nn.Module):
    """Residual convolutional block with pre-activation GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return h + residual


class AttentionBlock(nn.Module):
    """Spatial Self-Attention module used in the autoencoder bottleneck."""

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape

        norm_x = self.group_norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)

        q = self.to_q(norm_x).unsqueeze(1)
        k = self.to_k(norm_x).unsqueeze(1)
        v = self.to_v(norm_x).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v)
        out = self.to_out[0](out.squeeze(1))

        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return residual + out


class Downsampler2D(nn.Module):
    """Spatial downsampling layer applying asymmetric padding before 2D convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsampler2D(nn.Module):
    """Spatial upsampling via nearest interpolation followed by 2D convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetMidBlock2D(nn.Module):
    """Autoencoder bottleneck block."""

    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([AttentionBlock(channels, groups=groups)])
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, groups=groups),
            ResnetBlock2D(channels, channels, groups=groups),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2, add_downsample: bool = True) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsampler2D(out_channels)]) if add_downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, add_upsample: bool = True) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList([Upsampler2D(out_channels)]) if add_upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: FluxVAEConfig) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(cfg.in_channels, cfg.block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        current_ch = cfg.block_out_channels[0]
        for i, out_ch in enumerate(cfg.block_out_channels):
            is_final = i == len(cfg.block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    num_layers=cfg.layers_per_block,
                    add_downsample=not is_final,
                )
            )
            current_ch = out_ch

        self.mid_block = UNetMidBlock2D(cfg.block_out_channels[-1])
        self.conv_norm_out = nn.GroupNorm(32, cfg.block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(cfg.block_out_channels[-1], 2 * cfg.latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        return self.conv_out(self.conv_act(self.conv_norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, cfg: FluxVAEConfig) -> None:
        super().__init__()
        reversed_channels = list(reversed(cfg.block_out_channels))

        self.conv_in = nn.Conv2d(cfg.latent_channels, reversed_channels[0], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_channels[0])

        self.up_blocks = nn.ModuleList()
        current_ch = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels):
            is_final = i == len(reversed_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    num_layers=cfg.layers_per_block + 1,
                    add_upsample=not is_final,
                )
            )
            current_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(32, cfg.block_out_channels[0], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(cfg.block_out_channels[0], cfg.out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        return self.conv_out(self.conv_act(self.conv_norm_out(h)))


class AutoencoderKL(nn.Module):
    """Complete FLUX AutoencoderKL model."""

    def __init__(self, config: Optional[FluxVAEConfig] = None) -> None:
        super().__init__()
        self.config = config or FluxVAEConfig()
        self.scaling_factor = self.config.scaling_factor
        self.shift_factor = self.config.shift_factor
        self.latent_channels = self.config.latent_channels

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Applies FLUX shift and scaling: (z - shift) * scale."""
        return (latents - self.shift_factor) * self.scaling_factor

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverts FLUX normalization back to raw VAE space: (z / scale) + shift."""
        return (latents / self.scaling_factor) + self.shift_factor

    @torch.no_grad()
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        sample = self.decoder(z)
        if not return_dict:
            return sample
        return DecoderOutput(sample=sample)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> DecoderOutput:
        enc_output = self.encode(sample, return_dict=True)
        assert isinstance(enc_output, AutoencoderKLOutput)
        z = enc_output.latent_dist.sample(generator=generator) if sample_posterior else enc_output.latent_dist.mode()
        dec = self.decode(z, return_dict=True)
        assert isinstance(dec, DecoderOutput)
        return dec