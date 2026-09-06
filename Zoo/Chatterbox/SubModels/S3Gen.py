import math
from typing import Dict, List, Optional, Tuple, cast, Any
import numpy as np
from scipy.signal import get_window
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
from torch.nn.utils.parametrizations import weight_norm
from diffusers.models.attention_processor import Attention
from diffusers.models.activations import GELU
from einops import pack, rearrange, repeat

from Zoo.Chatterbox.SubModels.S3Tokenizer import S3Tokenizer

# =====================================================================
# 0. Mask & Audio Utilities
# =====================================================================
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    lengths = lengths.long()
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    return seq_range.unsqueeze(0).expand(batch_size, max_len) >= lengths.unsqueeze(-1)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    return (1.0 - mask.to(dtype)) * -1.0e+10


def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    return pos_idx.unsqueeze(0) < block_value.unsqueeze(1)


def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor, static_chunk_size: int = 0) -> torch.Tensor:
    if static_chunk_size > 0:
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, -1, xs.device).unsqueeze(0)
        return masks & chunk_masks
    return masks


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000
) -> torch.Tensor:
    if y.ndim == 1:
        y = y.unsqueeze(0)
    pad_val = (n_fft - hop_size) // 2
    y = F.pad(y.unsqueeze(1), (pad_val, pad_val), mode="reflect").squeeze(1)
    window = torch.hann_window(win_size).to(y.device)
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=window, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis_tensor = torch.from_numpy(mel_basis).float().to(y.device)
    return torch.log(torch.clamp(torch.matmul(mel_basis_tensor, spec), min=1e-5))


# =====================================================================
# 1. Speaker Encoder (CAMPPlus / X-Vector)
# =====================================================================
def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """Recreates exact layer names (batchnorm, relu) for strict=True matching."""
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
    return nonlinear


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class FCM(nn.Module):
    def __init__(self, feat_dim: int = 80, m_channels: int = 32):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = nn.Sequential(BasicResBlock(m_channels, m_channels, 2), BasicResBlock(m_channels, m_channels, 1))
        self.layer2 = nn.Sequential(BasicResBlock(m_channels, m_channels, 2), BasicResBlock(m_channels, m_channels, 1))
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class TDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.nonlinear = get_nonlinear("batchnorm-relu", out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // 2, 1)
        self.linear2 = nn.Conv1d(bn_channels // 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        seg = F.avg_pool1d(x, 100, 100, ceil_mode=True)
        shape = seg.shape
        pooled = seg.unsqueeze(-1).expand(*shape, 100).reshape(*shape[:-1], -1)[..., :x.shape[-1]]
        
        context = x.mean(-1, keepdim=True) + pooled
        m = torch.sigmoid(self.linear2(F.relu(self.linear1(context))))
        return y * m


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_nonlinear("batchnorm-relu", in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear("batchnorm-relu", bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x))))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers: int, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"tdnnd{i + 1}", CAMDenseTDNNLayer(in_channels + i * out_channels, out_channels, bn_channels, kernel_size, stride, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.nonlinear = get_nonlinear("batchnorm-relu", in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.nonlinear = get_nonlinear("batchnorm_", out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x.unsqueeze(-1))).squeeze(-1)


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim: int = 80, embedding_size: int = 192):
        super().__init__()
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        
        self.xvector = nn.Sequential()
        self.xvector.add_module("tdnn", TDNNLayer(channels, 128, 5, stride=2, dilation=1, padding=2))
        channels = 128
        
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            self.xvector.add_module(f"block{i + 1}", CAMDenseTDNNBlock(num_layers, channels, 32, 128, kernel_size, 1, dilation))
            channels += num_layers * 32
            self.xvector.add_module(f"transit{i + 1}", TransitLayer(channels, channels // 2))
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear("batchnorm-relu", channels))
        self.xvector.add_module("stats", nn.Identity())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x.permute(0, 2, 1))
        for name, layer in self.xvector.named_children():
            if name == "stats":
                mean = x.mean(dim=-1)
                std = x.std(dim=-1, unbiased=True)
                x = torch.cat([mean, std], dim=-1)
            else:
                x = layer(x)
        return x

    @torch.inference_mode()
    def inference(self, audio: torch.Tensor) -> torch.Tensor:
        # 1. Kaldi.fbank strictly requires audio to be on the CPU
        audio_cpu = audio.squeeze().cpu().float()
        
        # 2. Extract features on CPU
        features = [Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80) for au in [audio_cpu]]
        feature = features[0] - features[0].mean(dim=0, keepdim=True)
        
        # 3. Move the extracted features to the GPU (or whatever device the model is on)
        device = next(self.parameters()).device
        feature = feature.unsqueeze(0).to(dtype=torch.float32, device=device)
        
        # 4. Pass through the network
        return self.forward(feature)


# =====================================================================
# 2. Conformer Flow Token Encoder
# =====================================================================
class EspnetRelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.pe: Optional[torch.Tensor] = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            return
        pos = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe_pos = torch.zeros(x.size(1), self.d_model)
        pe_neg = torch.zeros(x.size(1), self.d_model)
        pe_pos[:, 0::2] = torch.sin(pos * div)
        pe_pos[:, 1::2] = torch.cos(pos * div)
        pe_neg[:, 0::2] = torch.sin(-pos * div)
        pe_neg[:, 1::2] = torch.cos(-pos * div)
        pe = torch.cat([torch.flip(pe_pos, [0]).unsqueeze(0), pe_neg[1:].unsqueeze(0)], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x)
        assert self.pe is not None
        pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)].to(x.device)
        return x * self.xscale, pos_emb


class LinearNoSubsampling(nn.Module):
    def __init__(self, idim: int = 512, odim: int = 512):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(idim, odim), nn.LayerNorm(odim, eps=1e-5), nn.Dropout(0.1))
        self.pos_enc = EspnetRelPositionalEncoding(odim)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask


class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int = 8, n_feat: int = 512):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(n_head, self.d_k))

def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = query.size()
        
        # 1. Project Q without transposing yet! Shape remains (B, T, h, d_k)
        q = self.linear_q(query).view(B, -1, self.h, self.d_k)
        
        # 2. Add bias to Q. Because q is (B, T, h, d_k) and pos_bias is (h, d_k),
        # PyTorch broadcasting matches perfectly. Then we transpose for Matmul.
        q_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2) # -> (B, h, T, d_k)
        q_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2) # -> (B, h, T, d_k)

        # 3. Process K, V, and P directly into transposed shapes
        k = self.linear_k(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(B, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)

        # 4. Matrix Multiplications
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))

        # 5. Shift & Masking
        zero_pad = torch.zeros((matrix_bd.size(0), matrix_bd.size(1), matrix_bd.size(2), 1), device=matrix_bd.device, dtype=matrix_bd.dtype)
        matrix_bd = torch.cat([zero_pad, matrix_bd], dim=-1)
        matrix_bd = matrix_bd.view(matrix_bd.size(0), matrix_bd.size(1), matrix_bd.size(3) + 1, matrix_bd.size(2))[:, :, 1:].view_as(matrix_bd)[:, :, :, :matrix_bd.size(-1) // 2 + 1]

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        if mask.size(2) > 0:
            scores = scores.masked_fill(mask.unsqueeze(1).eq(0)[:, :, :, :scores.size(-1)], -1e9)
            
        # 6. Apply Attention to V
        attn = torch.softmax(scores, dim=-1)
        x = (attn @ v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        
        return self.linear_out(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int = 512, hidden_units: int = 2048):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = nn.SiLU()
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(xs)))


class ConformerEncoderLayer(nn.Module):
    def __init__(self, size: int = 512):
        super().__init__()
        self.self_attn = RelPositionMultiHeadedAttention(n_head=8, n_feat=size)
        self.feed_forward = PositionwiseFeedForward(idim=size, hidden_units=2048)
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm_mha(x)
        x = residual + self.self_attn(x, x, x, mask, pos_emb)
        residual = x
        x = self.norm_ff(x)
        x = residual + self.feed_forward(x)
        return x, mask


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int = 512, pre_lookahead_len: int = 3):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs).transpose(1, 2).contiguous()
        return outputs + inputs


class ConformerUpsample1D(nn.Module):
    def __init__(self, channels: int = 512, out_channels: int = 512, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = self.conv(F.pad(outputs, (self.stride * 2, 0), value=0.0))
        return outputs, input_lengths * self.stride


class UpsampleConformerEncoder(nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 512):
        super().__init__()
        self.embed = LinearNoSubsampling(input_size, output_size)
        self.pre_lookahead_layer = PreLookaheadLayer(channels=output_size, pre_lookahead_len=3)
        self.encoders = nn.ModuleList([ConformerEncoderLayer(size=output_size) for _ in range(6)])
        self.up_layer = ConformerUpsample1D(channels=output_size, out_channels=output_size, stride=2)
        self.up_embed = LinearNoSubsampling(input_size, output_size)
        self.up_encoders = nn.ModuleList([ConformerEncoderLayer(size=output_size) for _ in range(4)])
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        chunk_masks = add_optional_chunk_mask(xs, masks, 0)
        xs = self.pre_lookahead_layer(xs)
        for layer in self.encoders:
            enc = cast(ConformerEncoderLayer, layer)
            xs, chunk_masks = enc(xs, chunk_masks, pos_emb)

        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()

        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        chunk_masks = add_optional_chunk_mask(xs, masks, 0)
        for layer in self.up_encoders:
            enc = cast(ConformerEncoderLayer, layer)
            xs, chunk_masks = enc(xs, chunk_masks, pos_emb)

        return self.after_norm(xs), masks


# =====================================================================
# 3. Flow Matching Decoder (Matcha / Diffusers Native Hierarchy)
# =====================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, self.dim0, self.dim1)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, self.causal_padding))


class CausalBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x * mask) * mask


class CausalResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([
            GELU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm3(hidden_states)
        return self.ff(norm_hidden_states) + hidden_states


class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels: int = 320, out_channels: int = 80, channels: Tuple[int, ...] = (256,), meanflow: bool = False):
        super().__init__()
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4  # 1024
        self.time_mlp = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # Down block
        resnet = CausalResnetBlock1D(dim=in_channels, dim_out=channels[0], time_emb_dim=time_embed_dim)
        transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim=channels[0], num_attention_heads=8, attention_head_dim=64) for _ in range(4)])
        downsample = CausalConv1d(channels[0], channels[0], 3)
        self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # Mid blocks (12)
        for _ in range(12):
            resnet = CausalResnetBlock1D(dim=channels[-1], dim_out=channels[-1], time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim=channels[-1], num_attention_heads=8, attention_head_dim=64) for _ in range(4)])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # Up block
        resnet = CausalResnetBlock1D(dim=channels[0] * 2, dim_out=channels[0], time_emb_dim=time_embed_dim)
        transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim=channels[0], num_attention_heads=8, attention_head_dim=64) for _ in range(4)])
        upsample = CausalConv1d(channels[0], channels[0], 3)
        self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = CausalBlock1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, spks: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None, r: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_embeddings(t).to(t.dtype)
        t_emb = self.time_mlp(t_emb)

        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for block in self.down_blocks:
            b_list = cast(nn.ModuleList, block)
            resnet_down = cast(CausalResnetBlock1D, b_list[0])
            tb_down = cast(nn.ModuleList, b_list[1])
            ds_down = cast(CausalConv1d, b_list[2])

            mask_down = masks[-1]
            x = resnet_down(x, mask_down, t_emb)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_to_bias(mask_down.bool() == 1, x.dtype)
            attn_mask = attn_mask.unsqueeze(1) # Changes (B, 1, T) to (B, 1, 1, T)
            for transformer_block in tb_down:
                tb = cast(BasicTransformerBlock, transformer_block)
                x = tb(x, attention_mask=attn_mask)
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = ds_down(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        for block in self.mid_blocks:
            b_list = cast(nn.ModuleList, block)
            resnet_mid = cast(CausalResnetBlock1D, b_list[0])
            tb_mid = cast(nn.ModuleList, b_list[1])

            x = resnet_mid(x, mask_mid, t_emb)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_to_bias(mask_mid.bool() == 1, x.dtype)
            for transformer_block in tb_mid:
                tb = cast(BasicTransformerBlock, transformer_block)
                x = tb(x, attention_mask=attn_mask)
            x = rearrange(x, "b t c -> b c t").contiguous()

        for block in self.up_blocks:
            b_list = cast(nn.ModuleList, block)
            resnet_up = cast(CausalResnetBlock1D, b_list[0])
            tb_up = cast(nn.ModuleList, b_list[1])
            us_up = cast(CausalConv1d, b_list[2])

            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet_up(x, mask_up, t_emb)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_to_bias(mask_up.bool() == 1, x.dtype)
            for transformer_block in tb_up:
                tb = cast(BasicTransformerBlock, transformer_block)
                x = tb(x, attention_mask=attn_mask)
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = us_up(x * mask_up)

        x = self.final_block(x, mask_up)
        return self.final_proj(x * mask_up) * mask


class CausalConditionalCFM(nn.Module):
    def __init__(self, in_channels: int = 240, spk_emb_dim: int = 80, estimator: Optional[ConditionalDecoder] = None):
        super().__init__()
        self.inference_cfg_rate = 0.7
        self.estimator = estimator

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert self.estimator is not None
        B, T = mu.size(0), x.size(2)
        x_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2 * B, 1, T], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2 * B], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2 * B, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)

        for t, r in zip(t_span[:-1], t_span[1:]):
            t_val = t.unsqueeze(dim=0)
            r_val = r.unsqueeze(dim=0)

            x_in[:B] = x_in[B:] = x
            mask_in[:B] = mask_in[B:] = mask
            mu_in[:B] = mu
            t_in[:B] = t_in[B:] = t_val
            if spks is not None:
                spks_in[:B] = spks
            if cond is not None:
                cond_in[:B] = cond

            dxdt = self.estimator(x=x_in, mask=mask_in, mu=mu_in, t=t_in, spks=spks_in, cond=cond_in)
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = (1.0 + self.inference_cfg_rate) * dxdt - self.inference_cfg_rate * cfg_dxdt
            x = x + (r_val - t_val) * dxdt

        return x

    @torch.inference_mode()
    def forward(self, mu: torch.Tensor, mask: torch.Tensor, n_timesteps: int, spks: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = torch.randn_like(mu)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.input_embedding = nn.Embedding(6561, 512)
        self.spk_embed_affine_layer = nn.Linear(192, 80)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(512, 80)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(self, token: torch.Tensor, token_len: torch.Tensor, prompt_token: torch.Tensor, prompt_token_len: torch.Tensor, prompt_feat: torch.Tensor, prompt_feat_len: Optional[torch.Tensor], embedding: torch.Tensor, finalize: bool = True, n_timesteps: int = 10) -> Tuple[torch.Tensor, None]:
        B = token.size(0)
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(token.long()) * mask

        h, h_masks = self.encoder(token, token_len)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        conds = torch.zeros([B, mel_len1 + mel_len2, 80], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=h_masks,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        return feat[:, :, mel_len1:], None


# =====================================================================
# 4. Vocoder (HiFTGenerator + F0 Predictor with Exact Weight Norms)
# =====================================================================
class ConvRNNF0Predictor(nn.Module):
    def __init__(self, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(cond_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.classifier(self.condnet(x).transpose(1, 2)).squeeze(-1))


class Snake(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            pad = int((kernel_size * dilation - dilation) / 2)
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=pad)))
            pad1 = int((kernel_size - 1) / 2)
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=pad1)))
        self.activations1 = nn.ModuleList([Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([Snake(channels) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class SineGen(nn.Module):
    def __init__(self, samp_rate: int = 24000, harmonic_num: int = 8):
        super().__init__()
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate

    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)), device=f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        phase_vec = (torch.rand(f0.size(0), self.harmonic_num + 1, 1, device=f0.device) * 2 * np.pi) - np.pi
        phase_vec[:, 0, :] = 0
        sine_waves = 0.1 * torch.sin(theta_mat + phase_vec)
        uv = (f0 > 10).float()
        noise = (uv * 0.003 + (1 - uv) * (0.1 / 3)) * torch.randn_like(sine_waves)
        return sine_waves * uv + noise, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int = 24000, harmonic_num: int = 8):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))
            sine_wavs = sine_wavs.transpose(1, 2)
            uv = uv.transpose(1, 2)
        sine_merge = torch.tanh(self.l_linear(sine_wavs.transpose(1, 2))).transpose(1, 2)
        noise = torch.randn_like(uv) * (0.1 / 3)
        return sine_merge, noise, uv


class HiFTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_source = SourceModuleHnNSF(24000, 8)
        self.f0_upsamp = nn.Upsample(scale_factor=480)
        self.f0_predictor = ConvRNNF0Predictor(80, 512)

        self.conv_pre = weight_norm(nn.Conv1d(80, 512, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(512, 256, 16, 8, padding=4)),
            weight_norm(nn.ConvTranspose1d(256, 128, 11, 5, padding=3)),
            weight_norm(nn.ConvTranspose1d(128, 64, 7, 3, padding=2))
        ])

        self.source_downs = nn.ModuleList([
            nn.Conv1d(18, 256, kernel_size=30, stride=15, padding=7),
            nn.Conv1d(18, 128, kernel_size=6, stride=3, padding=1),
            nn.Conv1d(18, 64, kernel_size=1, stride=1)
        ])
        self.source_resblocks = nn.ModuleList([
            ResBlock(256, 7, [1, 3, 5]),
            ResBlock(128, 7, [1, 3, 5]),
            ResBlock(64, 11, [1, 3, 5])
        ])

        self.resblocks = nn.ModuleList()
        for ch in [256, 128, 64]:
            for k in [3, 7, 11]:
                self.resblocks.append(ResBlock(ch, k, [1, 3, 5]))

        self.conv_post = weight_norm(nn.Conv1d(64, 18, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        stft_arr = np.asarray(get_window("hann", 16, fftbins=True), dtype=np.float32)
        self.stft_window = torch.from_numpy(stft_arr)

    def decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            s.squeeze(1), 16, 4, 16, window=self.stft_window.to(x.device),
            center=False, return_complex=True
        )
        spec = torch.view_as_real(spec)
        s_stft = torch.cat([spec[..., 0], spec[..., 1]], dim=1)

        x = self.conv_pre(x)
        for i in range(3):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            if i == 2:
                x = self.reflection_pad(x)

            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs: Optional[torch.Tensor] = None
            for j in range(3):
                block_out = self.resblocks[i * 3 + j](x)
                xs = block_out if xs is None else xs + block_out
            assert xs is not None
            x = xs / 3.0

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        mag = torch.clip(torch.exp(x[:, :9, :]), max=1e2)
        phase = torch.sin(x[:, 9:, :])
        real, img = mag * torch.cos(phase), mag * torch.sin(phase)
        wav = torch.istft(torch.complex(real, img), 16, 4, 16, window=self.stft_window.to(x.device))
        return torch.clamp(wav, -0.99, 0.99)

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor) -> torch.Tensor:
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        return self.decode(x=speech_feat, s=s)


# =====================================================================
# 5. Top-Level S3Gen Module
# =====================================================================
class S3Gen(nn.Module):
    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.speaker_encoder = CAMPPlus()

        encoder = UpsampleConformerEncoder(input_size=512, output_size=512)
        estimator = ConditionalDecoder(in_channels=320, out_channels=80, channels=(256,), meanflow=meanflow)
        decoder = CausalConditionalCFM(spk_emb_dim=80, estimator=estimator)
        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)

        self.mel2wav = HiFTGenerator()

        n_trim = 24000 // 50
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Overrides load_state_dict to safely inject the frozen tokenizer and generated buffers, allowing strict=True to pass."""
        model_dict = self.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("tokenizer.") and k != "trim_fade":
                filtered_state_dict[k] = v
                
        # Reinject tokenizer and buffer keys from the newly initialized model
        for k, v in model_dict.items():
            if k.startswith("tokenizer.") or k == "trim_fade":
                filtered_state_dict[k] = v
                
        return super().load_state_dict(filtered_state_dict, strict=strict)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.flow.parameters()).dtype

    def embed_ref(self, ref_wav: torch.Tensor, ref_sr: int) -> Dict[str, Any]:
        if ref_sr != 24000:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 24000)
        ref_wav_16 = torchaudio.functional.resample(ref_wav, 24000, 16000)

        ref_xvec = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))
        ref_mels = mel_spectrogram(ref_wav.to(self.device)).transpose(1, 2).to(dtype=self.dtype)
        ref_tokens, ref_lens = self.tokenizer(ref_wav_16.float().to(self.device))

        if ref_mels.shape[1] != 2 * ref_tokens.shape[1]:
            ref_tokens = ref_tokens[:, :ref_mels.shape[1] // 2]
            ref_lens[0] = ref_tokens.shape[1]

        return {
            "prompt_token": ref_tokens.to(self.device),
            "prompt_token_len": ref_lens.to(self.device),
            "prompt_feat": ref_mels.to(self.device),
            "prompt_feat_len": None,
            "embedding": ref_xvec.to(self.device)
        }

    @torch.inference_mode()
    def generate(
        self,
        speech_tokens: torch.Tensor,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: int = 24000,
        ref_dict: Optional[Dict[str, Any]] = None,
        n_cfm_timesteps: int = 10,
        skip_vocoder: bool = False
    ) -> torch.Tensor:
        if ref_dict is None:
            assert ref_wav is not None, "Must provide ref_wav or ref_dict"
            ref_dict = self.embed_ref(ref_wav, ref_sr)

        speech_tokens = torch.atleast_2d(speech_tokens).to(self.device)
        token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)

        mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=token_lens,
            finalize=True,
            n_timesteps=n_cfm_timesteps,
            **ref_dict
        )

        if skip_vocoder:
            return mels

        wav = self.mel2wav.inference(speech_feat=mels)
        trim_fade_tensor = cast(torch.Tensor, self.trim_fade)
        fade_len = trim_fade_tensor.size(0)
        wav[:, :fade_len] *= trim_fade_tensor.to(wav.device)
        return wav