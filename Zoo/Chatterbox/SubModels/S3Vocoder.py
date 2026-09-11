import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm


# ==========================================
# 1. Mel-Spectrogram Feature Extractor
# ==========================================
_MEL_BASIS = {}
_HANN_WINDOW = {}

def mel_spectrogram(y: torch.Tensor, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False) -> torch.Tensor:
    if y.ndim == 1:
        y = y.unsqueeze(0)

    key = f"{fmax}_{y.device}"
    if key not in _MEL_BASIS:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _MEL_BASIS[key] = torch.from_numpy(mel).float().to(y.device)
        _HANN_WINDOW[str(y.device)] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_HANN_WINDOW[str(y.device)], center=center, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True
    )
    spec = torch.view_as_real(spec)
    mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(_MEL_BASIS[key], mag)
    return torch.log(torch.clamp(mel_spec, min=1e-5))


# ==========================================
# 2. Neural Source Filter (NSF) Modules
# ==========================================
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    if "Conv" in m.__class__.__name__ and hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
        nn.init.normal_(m.weight, mean, std)


class Snake(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.view(1, -1, 1)
        return x + torch.sin(x * a).pow(2) / (a + 1e-9)


class ResBlock(nn.Module):
    def __init__(self, channels: int = 512, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilations
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([Snake(channels) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations1, self.activations2):
            x = x + c2(a2(c1(a1(x))))
        return x


class SineGen(nn.Module):
    def __init__(self, samp_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.sampling_rate = samp_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype).view(1, -1, 1)
        f_mat = f0 * harmonics / self.sampling_rate
        theta_mat = 2 * math.pi * (torch.cumsum(f_mat, dim=-1) % 1)
        phase = (torch.rand(f0.size(0), self.harmonic_num + 1, 1, device=f0.device, dtype=f0.dtype) * 2 - 1) * math.pi
        phase[:, 0, :] = 0
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase)
        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.noise_std + (1.0 - uv) * (self.sine_amp / 3.0)
        noise = noise_amp * torch.randn_like(sine_waves)
        return sine_waves * uv + noise, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, upsample_scale: int, harmonic_num: int = 0, sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))
            sine_wavs, uv = sine_wavs.transpose(1, 2), uv.transpose(1, 2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * (self.sine_amp / 3.0)
        return sine_merge, noise, uv


class ConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(),
        )
        self.classifier = nn.Linear(cond_channels, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x).transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


# ==========================================
# 3. HiFTGenerator Vocoder Backbone
# ==========================================
class HiFTGenerator(nn.Module):
    stft_window: torch.Tensor

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 5, 3],
        upsample_kernel_sizes: List[int] = [16, 11, 7],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        total_upsample = int(np.prod(upsample_rates)) * istft_params["hop_len"]
        self.m_source = SourceModuleHnNSF(sampling_rate, total_upsample, nb_harmonics, nsf_alpha, nsf_sigma, nsf_voiced_threshold)
        self.f0_upsamp = nn.Upsample(scale_factor=total_upsample)
        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))

        self.ups = nn.ModuleList([
            weight_norm(ConvTranspose1d(base_channels // (2**i), base_channels // (2**(i + 1)), k, u, padding=(k - u) // 2))
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        ])

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_cum_rates = np.cumprod([1] + upsample_rates[::-1][:-1])[::-1]

        for i, (u, k, d) in enumerate(zip(downsample_cum_rates, source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            ch = base_channels // (2 ** (i + 1))
            stride_down = Conv1d(istft_params["n_fft"] + 2, ch, 1, 1) if u == 1 else Conv1d(istft_params["n_fft"] + 2, ch, u * 2, u, padding=u // 2)
            self.source_downs.append(stride_down)
            self.source_resblocks.append(ResBlock(ch, k, d))

        self.resblocks = nn.ModuleList([
            ResBlock(base_channels // (2 ** (i + 1)), k, d)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        final_ch = base_channels // (2 ** len(self.ups))
        self.conv_post = weight_norm(Conv1d(final_ch, istft_params["n_fft"] + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.register_buffer("stft_window", torch.hann_window(istft_params["n_fft"], periodic=True), persistent=False)
        self.f0_predictor = f0_predictor

    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(x, n_fft=self.istft_params["n_fft"], hop_length=self.istft_params["hop_len"], win_length=self.istft_params["n_fft"], window=self.stft_window, return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        mag = torch.clamp(magnitude, max=1e2)
        c = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        return torch.istft(c, n_fft=self.istft_params["n_fft"], hop_length=self.istft_params["hop_len"], win_length=self.istft_params["n_fft"], window=self.stft_window)

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        s_real, s_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, self.lrelu_slope))
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            si = self.source_resblocks[i](self.source_downs[i](s_stft))
            x = x + si

            res_start = i * self.num_kernels
            xs: Optional[torch.Tensor] = None
            for j in range(self.num_kernels):
                out = self.resblocks[res_start + j](x)
                xs = out if xs is None else xs + out
            assert xs is not None
            x = xs / self.num_kernels

        x = self.conv_post(F.leaky_relu(x))
        n_fft_bins = self.istft_params["n_fft"] // 2 + 1
        magnitude = torch.exp(x[:, :n_fft_bins, :])
        phase = torch.sin(x[:, n_fft_bins:, :])
        x = self._istft(magnitude, phase)
        return torch.clamp(x, -self.audio_limit, self.audio_limit)

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = torch.zeros(1, 1, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.f0_predictor is None:
            raise RuntimeError("f0_predictor must be set on HiFTGenerator to run inference.")
        
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if cache_source.shape[2] != 0:
            s[:, :, :cache_source.shape[2]] = cache_source
        return self.decode(x=speech_feat, s=s), s