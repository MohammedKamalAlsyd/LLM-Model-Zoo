import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Helper Data Classes & Distributions
# ==============================================================================
@dataclass
class DecoderOutput:
    sample: torch.Tensor

class DiagonalGaussianDistribution:
    """Represents a diagonal Gaussian distribution parameterized by mean and logvar."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

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

# ==============================================================================
# Core Architectural Modules
# ==============================================================================

class ResnetBlock2D(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

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
    """Spatial Self-Attention used in mid_block (1 head, head_dim = channels)."""

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape

        norm_x = self.group_norm(x)
        norm_x = norm_x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]

        q = self.to_q(norm_x).unsqueeze(1)  # [B, 1, HW, C]
        k = self.to_k(norm_x).unsqueeze(1)  # [B, 1, HW, C]
        v = self.to_v(norm_x).unsqueeze(1)  # [B, 1, HW, C]

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)  # [B, 1, HW, C]
        out = out.squeeze(1)  # [B, HW, C]
        out = self.to_out[0](out)

        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        return residual + out


class Downsampler2D(nn.Module):
    """Downsampling layer with asymmetric padding: right=1, bottom=1."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad (left=0, right=1, top=0, bottom=1) to ensure exact halving
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsampler2D(nn.Module):
    """Nearest interpolation followed by a 3x3 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetMidBlock2D(nn.Module):
    """Mid-block consisting of ResNet -> Self-Attention -> ResNet."""

    def __init__(self, channels: int, groups: int = 32):
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


# ==============================================================================
# Down & Up Stages
# ==============================================================================

class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2, add_downsample: bool = True):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(ch_in, out_channels))
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsampler2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, add_upsample: bool = True):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(ch_in, out_channels))
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsampler2D(out_channels)])
        else:
            self.upsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


# ==============================================================================
# Encoder & Decoder
# ==============================================================================

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        current_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            is_final = i == len(block_out_channels) - 1
            block = DownEncoderBlock2D(
                in_channels=current_ch,
                out_channels=out_ch,
                num_layers=layers_per_block,
                add_downsample=not is_final,
            )
            self.down_blocks.append(block)
            current_ch = out_ch

        self.mid_block = UNetMidBlock2D(block_out_channels[-1])
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = self.conv_out(self.conv_act(self.conv_norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))  # [512, 512, 256, 128]

        self.conv_in = nn.Conv2d(in_channels, reversed_channels[0], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_channels[0])

        self.up_blocks = nn.ModuleList()
        current_ch = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels):
            is_final = i == len(reversed_channels) - 1
            block = UpDecoderBlock2D(
                in_channels=current_ch,
                out_channels=out_ch,
                num_layers=layers_per_block + 1,
                add_upsample=not is_final,
            )
            self.up_blocks.append(block)
            current_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_out(self.conv_act(self.conv_norm_out(h)))
        return h


# ==============================================================================
# AutoencoderKL Model Definition
# ==============================================================================

class AutoencoderKL(nn.Module):
    r"""
    FLUX.1 Variational Autoencoder (VAE) architecture with 16 latent channels,
    scale factor 0.3611, and shift factor 0.1159.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        scaling_factor: float = 0.3611,
        shift_factor: float = 0.1159,
        sample_size: int = 1024,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.latent_channels = latent_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        # Tiling attributes
        self.use_tiling = False
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // (2 ** (len(block_out_channels) - 1))
        self.tile_overlap_factor = 0.25

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.use_tiling = False

    # --------------------------------------------------------------------------
    # Flux Latent Scaling / Shifting
    # --------------------------------------------------------------------------
    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized representation used by FLUX Transformer."""
        return (latents - self.shift_factor) * self.scaling_factor

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Revert back to VAE latent scale for decoding."""
        return (latents / self.scaling_factor) + self.shift_factor

    # --------------------------------------------------------------------------
    # Encode & Decode Pass
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            moments = self._tiled_encode(x)
        else:
            moments = self.encoder(x)

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
        ):
            sample = self._tiled_decode(z)
        else:
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
        posterior = enc_output.latent_dist

        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z, return_dict=True)
        assert isinstance(dec, DecoderOutput)
        return dec

    # --------------------------------------------------------------------------
    # Tiling Helpers (For large images)
    # --------------------------------------------------------------------------
    @staticmethod
    def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            weight = y / blend_extent
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - weight) + b[:, :, y, :] * weight
        return b

    @staticmethod
    def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            weight = x / blend_extent
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - weight) + b[:, :, :, x] * weight
        return b

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                row.append(self.encoder(tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        return torch.cat(result_rows, dim=2)

    def _tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                row.append(self.decoder(tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        return torch.cat(result_rows, dim=2)