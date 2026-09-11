import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio.compliance.kaldi as Kaldi
from diffusers.models.attention_processor import Attention
from einops import pack, rearrange, repeat


# ==========================================
# 1. Mask Utilities
# ==========================================
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    lengths = lengths.long()
    resolved_max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(0, resolved_max_len, dtype=torch.int64, device=lengths.device)
    return seq_range.unsqueeze(0).expand(lengths.size(0), resolved_max_len) >= lengths.unsqueeze(-1)


def subsequent_chunk_mask(size: int, chunk_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode="trunc") + 1) * chunk_size
    return pos_idx.unsqueeze(0) < block_value.unsqueeze(1)


def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor, static_chunk_size: int = 0) -> torch.Tensor:
    if static_chunk_size > 0:
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, xs.device).unsqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks
    if (chunk_masks.sum(dim=-1) == 0).any():
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks


def get_chunk_mask(mask: torch.Tensor, seq_len: int, chunk_size: int = 0) -> torch.Tensor:
    if chunk_size > 0:
        pos = torch.arange(seq_len, device=mask.device)
        block = (torch.div(pos, chunk_size, rounding_mode="trunc") + 1) * chunk_size
        return mask.bool() & (pos.unsqueeze(0) < block.unsqueeze(1)).unsqueeze(0)
    return mask.bool()


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (1.0 - mask.to(dtype)) * -1e10


# ==========================================
# 2. CAMPPlus Speaker Encoder (X-Vector)
# ==========================================
def extract_feature(audio: torch.Tensor):
    fbanks = [Kaldi.fbank(a.unsqueeze(0), num_mel_bins=80) for a in audio]
    fbanks = [f - f.mean(dim=0, keepdim=True) for f in fbanks]
    lens = [f.shape[0] for f in fbanks]
    times = [a.shape[0] for a in audio]
    padded = torch.zeros(len(fbanks), max(lens), fbanks[0].shape[1], dtype=fbanks[0].dtype, device=fbanks[0].device)
    for i, f in enumerate(fbanks):
        padded[i, :f.shape[0]] = f
    return padded, lens, times


def get_nonlinear(config_str: str, channels: int):
    seq = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            seq.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            seq.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            seq.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            seq.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return seq


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        layers += [block(self.in_planes, planes, 1) for _ in range(num_blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        x = self.layer2(self.layer1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        return x.flatten(1, 2)


class StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int, bias: bool, reduction: int = 2):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = self.relu(self.linear1(x.mean(-1, keepdim=True) + self.seg_pooling(x)))
        return y * self.sigmoid(self.linear2(context))

    def seg_pooling(self, x: torch.Tensor, seg_len: int = 100, stype: str = "avg") -> torch.Tensor:
        pool_fn = F.avg_pool1d if stype == "avg" else F.max_pool1d
        seg = pool_fn(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        return seg.repeat_interleave(seg_len, dim=-1)[..., :x.shape[-1]]


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False, config_str: str = "batchnorm-relu", memory_efficient: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear1(self.nonlinear1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bn = cp.checkpoint(self.bn_function, x) if (self.training and self.memory_efficient) else self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x_bn))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers: int, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False, config_str: str = "batchnorm-relu", memory_efficient: bool = False):
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                f"tdnnd{i + 1}",
                CAMDenseTDNNLayer(
                    in_channels=in_channels + i * out_channels,
                    out_channels=out_channels,
                    bn_channels=bn_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x.unsqueeze(-1)).squeeze(-1) if x.ndim == 2 else self.linear(x)
        return self.nonlinear(x)


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim: int = 80, embedding_size: int = 192, growth_rate: int = 32, bn_size: int = 4, init_channels: int = 128, config_str: str = "batchnorm-relu", memory_efficient: bool = False, output_level: str = "segment"):
        super().__init__()
        self.head = FCM(feat_dim=feat_dim)
        self.output_level = output_level
        channels = self.head.out_channels

        self.xvector = nn.Sequential()
        self.xvector.add_module("tdnn", TDNNLayer(channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str))
        channels = init_channels

        for i, (num_layers, k_size, dil) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            self.xvector.add_module(f"block{i + 1}", CAMDenseTDNNBlock(num_layers, channels, growth_rate, bn_size * growth_rate, k_size, dilation=dil, config_str=config_str, memory_efficient=memory_efficient))
            channels += num_layers * growth_rate
            self.xvector.add_module(f"transit{i + 1}", TransitLayer(channels, channels // 2, bias=False, config_str=config_str))
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))
        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xvector(self.head(x.permute(0, 2, 1)))
        return x.transpose(1, 2) if self.output_level == "frame" else x

    def inference(self, audio_list: torch.Tensor) -> torch.Tensor:
        speech, _, _ = extract_feature(audio_list)
        return self.forward(speech.to(torch.float32))


# ==========================================
# 3. Conformer Upsampling Blocks
# ==========================================
class EspnetRelPositionalEncoding(nn.Module):
    pe: Optional[torch.Tensor]

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe_pos = torch.zeros(x.size(1), self.d_model)
        pe_neg = torch.zeros(x.size(1), self.d_model)
        pos = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe_pos[:, 0::2] = torch.sin(pos * div)
        pe_pos[:, 1::2] = torch.cos(pos * div)
        pe_neg[:, 0::2] = torch.sin(-1 * pos * div)
        pe_neg[:, 1::2] = torch.cos(-1 * pos * div)
        pe = torch.cat([torch.flip(pe_pos, [0]).unsqueeze(0), pe_neg[1:].unsqueeze(0)], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        assert self.pe is not None
        return self.pe[:, self.pe.size(1) // 2 - size + 1: self.pe.size(1) // 2 + size]

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x)
        return self.dropout(x * self.xscale), self.dropout(self.position_encoding(offset=offset, size=x.size(1)))


class LinearNoSubsampling(nn.Module):
    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc: nn.Module):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(idim, odim), nn.LayerNorm(odim, eps=1e-5), nn.Dropout(dropout_rate))
        self.pos_enc = pos_enc

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, 0)
        return x, pos_emb, x_mask


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        return x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        B, T = query.size(0), query.size(1)
        q = self.linear_q(query).view(B, -1, self.h, self.d_k)
        k = self.linear_k(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(B, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)

        q_with_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        if mask.size(2) > 0:
            m = mask.unsqueeze(1).eq(0)[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(m, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(self.dropout(attn), v)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linear_out(x)


class ConformerEncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm_mha(x), self.norm_mha(x), self.norm_mha(x), mask, pos_emb))
        return x + self.dropout(self.feed_forward(self.norm_ff(x)))


class ConformerUpsample1D(nn.Module):
    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        return self.conv(outputs), input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 3):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs.transpose(1, 2).contiguous()
        out = F.pad(out, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        out = F.leaky_relu(self.conv1(out))
        out = F.pad(out, (2, 0), mode="constant", value=0.0)
        return (self.conv2(out).transpose(1, 2).contiguous()) + inputs


class UpsampleConformerEncoder(nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 512, attention_heads: int = 8, linear_units: int = 2048, num_blocks: int = 6, dropout_rate: float = 0.1, positional_dropout_rate: float = 0.1, attention_dropout_rate: float = 0.1):
        super().__init__()
        self._output_size = output_size
        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, EspnetRelPositionalEncoding(output_size, positional_dropout_rate))
        self.pre_lookahead_layer = PreLookaheadLayer(channels=output_size, pre_lookahead_len=3)
        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(output_size, RelPositionMultiHeadedAttention(attention_heads, output_size, attention_dropout_rate), PositionwiseFeedForward(output_size, linear_units, dropout_rate), dropout_rate)
            for _ in range(num_blocks)
        ])
        self.up_layer = ConformerUpsample1D(channels=output_size, out_channels=output_size, stride=2)
        self.up_embed = LinearNoSubsampling(input_size, output_size, dropout_rate, EspnetRelPositionalEncoding(output_size, positional_dropout_rate))
        self.up_encoders = nn.ModuleList([
            ConformerEncoderLayer(output_size, RelPositionMultiHeadedAttention(attention_heads, output_size, attention_dropout_rate), PositionwiseFeedForward(output_size, linear_units, dropout_rate), dropout_rate)
            for _ in range(4)
        ])
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        chunk_masks = add_optional_chunk_mask(xs, masks)
        xs = self.pre_lookahead_layer(xs)
        for layer in self.encoders:
            xs = layer(xs, chunk_masks, pos_emb)

        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()

        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        chunk_masks = add_optional_chunk_mask(xs, masks)
        for layer in self.up_encoders:
            xs = layer(xs, chunk_masks, pos_emb)

        return self.after_norm(xs), masks


# ==========================================
# 4. Conditional Decoder U-Net (Flow Matching Backbone)
# ==========================================
class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, self.dim0, self.dim1)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, self.causal_padding))


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), nn.Mish())

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x * mask) * mask


class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super().__init__(dim, dim_out)
        self.block = nn.Sequential(CausalConv1d(dim, dim_out, 3), Transpose(1, 2), nn.LayerNorm(dim_out), Transpose(1, 2), nn.Mish())


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask) + self.mlp(time_emb).unsqueeze(-1)
        return self.block2(h, mask) + self.res_conv(x * mask)


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderUpsample1D(nn.Module):
    def __init__(self, channels: int, use_conv_transpose: bool = True, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        resolved_out = out_channels or channels
        self.use_conv_transpose = use_conv_transpose
        self.conv = nn.ConvTranspose1d(channels, resolved_out, 4, 2, 1) if use_conv_transpose else nn.Conv1d(channels, resolved_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) if self.use_conv_transpose else self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate=self.approximate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0, activation_fn: str = "gelu", final_dropout: bool = False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        act_fn = GELU(dim, inner_dim, approximate="tanh" if activation_fn == "gelu-approximate" else "none")
        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)])
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, dropout: float = 0.0, activation_fn: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=False, upcast_attention=False)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn1(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = attn_out + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels: int = 320, out_channels: int = 80, causal: bool = True, channels: Union[Tuple[int, ...], List[int]] = [256], dropout: float = 0.0, attention_head_dim: int = 64, n_blocks: int = 4, num_mid_blocks: int = 12, num_heads: int = 8, act_fn: str = "gelu", meanflow: bool = False):
        super().__init__()
        channels = tuple(channels)
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.static_chunk_size = 0

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim, act_fn="silu")

        # Down Blocks
        self.down_blocks = nn.ModuleList()
        output_channel = in_channels
        for i, ch in enumerate(channels):
            is_last = i == len(channels) - 1
            resnet = (CausalResnetBlock1D if self.causal else ResnetBlock1D)(dim=output_channel, dim_out=ch, time_emb_dim=time_embed_dim)
            transformers = nn.ModuleList([BasicTransformerBlock(ch, num_heads, attention_head_dim, dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)])
            downsample = Downsample1D(ch) if not is_last else (CausalConv1d(ch, ch, 3) if self.causal else nn.Conv1d(ch, ch, 3, padding=1))
            self.down_blocks.append(nn.ModuleList([resnet, transformers, downsample]))
            output_channel = ch

        # Mid Blocks
        self.mid_blocks = nn.ModuleList()
        mid_ch = channels[-1]
        for _ in range(num_mid_blocks):
            resnet = (CausalResnetBlock1D if self.causal else ResnetBlock1D)(mid_ch, mid_ch, time_embed_dim)
            transformers = nn.ModuleList([BasicTransformerBlock(mid_ch, num_heads, attention_head_dim, dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)])
            self.mid_blocks.append(nn.ModuleList([resnet, transformers]))

        # Up Blocks
        self.up_blocks = nn.ModuleList()
        up_channels = channels[::-1] + (channels[0],)
        for i in range(len(up_channels) - 1):
            in_ch = up_channels[i] * 2
            out_ch = up_channels[i + 1]
            is_last = i == len(up_channels) - 2

            resnet = (CausalResnetBlock1D if self.causal else ResnetBlock1D)(in_ch, out_ch, time_embed_dim)
            transformers = nn.ModuleList([BasicTransformerBlock(out_ch, num_heads, attention_head_dim, dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)])
            upsample = DecoderUpsample1D(out_ch, use_conv_transpose=True) if not is_last else (CausalConv1d(out_ch, out_ch, 3) if self.causal else nn.Conv1d(out_ch, out_ch, 3, padding=1))
            self.up_blocks.append(nn.ModuleList([resnet, transformers, upsample]))

        self.final_block = CausalBlock1D(up_channels[-1], up_channels[-1]) if self.causal else Block1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], self.out_channels, 1)

    @property
    def dtype(self):
        return self.final_proj.weight.dtype

    def _run_transformer(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        x = rearrange(x, "b c t -> b t c").contiguous()
        chunk_mask = get_chunk_mask(mask, x.shape[1], self.static_chunk_size)
        attn_mask = mask_to_bias(chunk_mask == 1, x.dtype)
        for tb in blocks:
            x = tb(hidden_states=x, attention_mask=attn_mask, timestep=t)
        return rearrange(x, "b t c -> b c t").contiguous()

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        t_emb = self.time_mlp(self.time_embeddings(t).to(t.dtype))

        cond_list = [x, mu]
        if spks is not None:
            cond_list.append(repeat(spks, "b c -> b c t", t=x.shape[-1]))
        if cond is not None:
            cond_list.append(cond)
        x = pack(cond_list, "b * t")[0]

        hiddens: List[torch.Tensor] = []
        masks: List[torch.Tensor] = [mask]

        # Down Blocks
        for block in self.down_blocks:
            b = cast(List[Any], block)
            resnet, transformers, downsample = b[0], cast(nn.ModuleList, b[1]), b[2]
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)
            x = self._run_transformer(x, mask_down, t_emb, transformers)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid Blocks
        for block in self.mid_blocks:
            b = cast(List[Any], block)
            resnet, transformers = b[0], cast(nn.ModuleList, b[1])
            x = resnet(x, mask_mid, t_emb)
            x = self._run_transformer(x, mask_mid, t_emb, transformers)

        # Up Blocks
        for block in self.up_blocks:
            b = cast(List[Any], block)
            resnet, transformers, upsample = b[0], cast(nn.ModuleList, b[1]), b[2]
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t_emb)
            x = self._run_transformer(x, mask_up, t_emb, transformers)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        return self.final_proj(x * mask_up) * mask


# ==========================================
# 5. CFM Solver & Flow Wrapper
# ==========================================
class CausalConditionalCFM(nn.Module):
    def __init__(self, in_channels: int = 240, spk_emb_dim: int = 80, estimator: Any = None):
        super().__init__()
        self.n_feats = in_channels
        self.spk_emb_dim = spk_emb_dim
        self.solver = "euler"
        self.sigma_min = 1e-6
        self.t_scheduler = "cosine"
        self.inference_cfg_rate = 0.7
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu: torch.Tensor, mask: torch.Tensor, n_timesteps: int, temperature: float = 1.0, spks: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None, noised_mels: Optional[torch.Tensor] = None, meanflow: bool = False):
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if not meanflow and self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Euler solver with CFG
        B, T = mu.size(0), z.size(2)
        x_in = torch.zeros([2 * B, 80, T], device=z.device, dtype=z.dtype)
        mask_in = torch.zeros([2 * B, 1, T], device=z.device, dtype=z.dtype)
        mu_in = torch.zeros([2 * B, 80, T], device=z.device, dtype=z.dtype)
        t_in = torch.zeros([2 * B], device=z.device, dtype=z.dtype)
        spks_in = torch.zeros([2 * B, 80], device=z.device, dtype=z.dtype)
        cond_in = torch.zeros([2 * B, 80, T], device=z.device, dtype=z.dtype)

        mask_in[:B] = mask_in[B:] = mask
        mu_in[:B] = mu
        if spks is not None:
            spks_in[:B] = spks
        if cond is not None:
            cond_in[:B] = cond

        x = z
        for t, r in zip(t_span[:-1], t_span[1:]):
            x_in[:B] = x_in[B:] = x
            t_in[:B] = t_in[B:] = t

            dxdt = self.estimator(x=x_in, mask=mask_in, mu=mu_in, t=t_in, spks=spks_in, cond=cond_in)
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = (1.0 + self.inference_cfg_rate) * dxdt - self.inference_cfg_rate * cfg_dxdt
            x = x + (r - t) * dxdt

        return x, None


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 80, spk_embed_dim: int = 192, vocab_size: int = 6561, encoder: Any = None, decoder: Any = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        encoder_out_size = self.encoder.output_size() if hasattr(self.encoder, "output_size") else input_size
        self.encoder_proj = nn.Linear(encoder_out_size, output_size)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(self, token: torch.Tensor, token_len: torch.Tensor, prompt_token: torch.Tensor, prompt_token_len: torch.Tensor, prompt_feat: torch.Tensor, embedding: torch.Tensor, finalize: bool, n_timesteps: int = 10, **kwargs):
        B = token.size(0)
        embedding = self.spk_embed_affine_layer(F.normalize(torch.atleast_2d(embedding), dim=1))

        token = torch.cat([prompt_token, token], dim=1)
        token_len = prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(token.long()) * mask

        h, h_masks = self.encoder(token, token_len)
        if not finalize:
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]

        h = self.encoder_proj(h)
        h_lengths = h_masks.sum(dim=-1).squeeze(-1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros([B, self.output_size, mel_len1 + mel_len2], device=token.device, dtype=h.dtype)
        conds[:, :, :mel_len1] = prompt_feat.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)
        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)

        feat, _ = self.decoder(mu=h.transpose(1, 2).contiguous(), mask=mask, spks=embedding, cond=conds, n_timesteps=n_timesteps)
        feat = feat[:, :, mel_len1:]
        return feat, None