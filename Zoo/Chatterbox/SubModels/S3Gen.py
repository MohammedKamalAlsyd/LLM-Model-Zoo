import math
from typing import Dict, List, Optional, Tuple, cast
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import numpy as np
from torch.nn.utils.parametrizations import weight_norm
from diffusers.models.attention_processor import Attention
from Zoo.Chatterbox.SubModels.S3Tokenizer import S3Tokenizer


# ==========================================
# 0. Local Audio Utils
# ==========================================
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
    """Fast, accurate Mel-Spectrogram extraction matching original Matcha-TTS."""
    if y.ndim == 1:
        y = y.unsqueeze(0)
    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect").squeeze(1)
    
    window = torch.hann_window(win_size).to(y.device)
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=window, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis_tensor = torch.from_numpy(mel_basis).float().to(y.device)
    spec = torch.matmul(mel_basis_tensor, spec)
    return torch.log(torch.clamp(spec, min=1e-5))


# ==========================================
# 1. Speaker Encoder (CAMPPlus)
# ==========================================
class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride=(stride, 1), bias=False),
            nn.BatchNorm2d(planes)
        ) if stride != 1 or in_planes != planes else nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, stride: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = nn.Sequential(nn.BatchNorm1d(in_channels), nn.ReLU(inplace=True))
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = nn.Sequential(nn.BatchNorm1d(bn_channels), nn.ReLU(inplace=True))
        
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.linear1_cam = nn.Conv1d(bn_channels, bn_channels // 2, 1)
        self.linear2_cam = nn.Conv1d(bn_channels // 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.nonlinear2(self.linear1(self.nonlinear1(x)))
        y = self.linear_local(h)
        pooled = F.avg_pool1d(h, 100, 100, ceil_mode=True).unsqueeze(-1).expand(*h.shape[:-1], 100).reshape(*h.shape[:-1], -1)[..., :h.shape[-1]]
        context = h.mean(-1, keepdim=True) + pooled
        m = torch.sigmoid(self.linear2_cam(F.relu(self.linear1_cam(context))))
        return y * m


class CAMPPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            BasicResBlock(32, 32, 2), BasicResBlock(32, 32, 1),
            BasicResBlock(32, 32, 2), BasicResBlock(32, 32, 1),
            nn.Conv2d(32, 32, 3, (2, 1), 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.xvector = nn.ModuleList([
            nn.Conv1d(320, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(),
            *[CAMDenseTDNNLayer(128 + i * 32, 32, 128, 3, 1, 1) for i in range(12)],
            nn.Conv1d(512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            *[CAMDenseTDNNLayer(256 + i * 32, 32, 128, 3, 1, 2) for i in range(24)],
            nn.Conv1d(1024, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            *[CAMDenseTDNNLayer(512 + i * 32, 32, 128, 3, 1, 2) for i in range(16)],
            nn.Conv1d(1024, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 3072, 1)
        ])
        self.stats = nn.Linear(6144, 192)

    @torch.inference_mode()
    def inference(self, audio: torch.Tensor) -> torch.Tensor:
        x = Kaldi.fbank(audio.squeeze(), num_mel_bins=80)
        x = (x - x.mean(dim=0, keepdim=True)).unsqueeze(0).transpose(1, 2).unsqueeze(1)
        x = self.head(x).flatten(1, 2)
        for layer in self.xvector:
            x = torch.cat([x, layer(x)], dim=1) if isinstance(layer, CAMDenseTDNNLayer) else layer(x)
        stats = torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)
        return self.stats(stats)


# ==========================================
# 2. Flow Encoder (Token -> Hidden States)
# ==========================================
class EspnetRelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model, self.xscale = d_model, math.sqrt(d_model)
        self.pe: Optional[torch.Tensor] = None

    def extend_pe(self, length: int, device: torch.device):
        if self.pe is not None and self.pe.size(1) >= length * 2 - 1:
            return
        pos = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / self.d_model))
        pe_pos = torch.zeros(length, self.d_model, device=device)
        pe_neg = torch.zeros(length, self.d_model, device=device)
        pe_pos[:, 0::2], pe_pos[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        pe_neg[:, 0::2], pe_neg[:, 1::2] = torch.sin(-pos * div), torch.cos(-pos * div)
        self.pe = torch.cat([torch.flip(pe_pos, [0]).unsqueeze(0), pe_neg[1:].unsqueeze(0)], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x.size(1), x.device)
        assert self.pe is not None
        pe_slice = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)]
        return x * self.xscale, pe_slice


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

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.linear_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)

        q_u, q_v = (q + self.pos_bias_u).transpose(1, 2), (q + self.pos_bias_v).transpose(1, 2)
        ac, bd = torch.matmul(q_u, k.transpose(-2, -1)), torch.matmul(q_v, p.transpose(-2, -1))
        
        bd = F.pad(bd, (1, 0)).view(B, self.h, bd.size(3) + 1, bd.size(2))[:, :, 1:].view_as(bd)[:, :, :, :bd.size(-1) // 2 + 1]
        
        scores = (ac + bd) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        out = torch.softmax(scores, dim=-1) @ v
        return self.linear_out(out.transpose(1, 2).reshape(B, T, -1))


class UpsampleConformerEncoder(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim, eps=1e-5))
        self.pos_enc1 = EspnetRelPositionalEncoding(dim)
        self.pre_lookahead = nn.Sequential(nn.Conv1d(dim, dim, 4), nn.LeakyReLU(), nn.Conv1d(dim, dim, 3))
        
        def _make_layer():
            return nn.ModuleList([
                nn.LayerNorm(dim, eps=1e-12),
                RelPositionMultiHeadedAttention(),
                nn.LayerNorm(dim, eps=1e-12),
                nn.Sequential(nn.Linear(dim, 2048), nn.SiLU(), nn.Linear(2048, dim))
            ])
            
        self.encoders = nn.ModuleList([_make_layer() for _ in range(6)])
        self.up_layer = nn.Conv1d(dim, dim, 5, padding=2)
        self.up_embed = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim, eps=1e-5))
        self.pos_enc2 = EspnetRelPositionalEncoding(dim)
        self.up_encoders = nn.ModuleList([_make_layer() for _ in range(4)])
        self.after_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, xs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, pos_emb = self.pos_enc1(self.embed(xs))
        
        h = F.pad(xs.transpose(1, 2), (0, 3))
        h = F.pad(self.pre_lookahead[1](self.pre_lookahead[0](h)), (2, 0))
        xs = xs + self.pre_lookahead[2](h).transpose(1, 2)
        
        for layer in self.encoders:
            layer_mods = cast(nn.ModuleList, layer)
            norm1, attn, norm2, ffn = layer_mods[0], layer_mods[1], layer_mods[2], layer_mods[3]
            xs = xs + attn(norm1(xs), pos_emb, mask)
            xs = xs + ffn(norm2(xs))
            
        xs = F.interpolate(xs.transpose(1, 2), scale_factor=2.0, mode="nearest")
        xs = self.up_layer(F.pad(xs, (4, 0))).transpose(1, 2)
        mask = mask.repeat_interleave(2, dim=-1)

        xs, pos_emb = self.pos_enc2(self.up_embed(xs))
        for layer in self.up_encoders:
            layer_mods = cast(nn.ModuleList, layer)
            norm1, attn, norm2, ffn = layer_mods[0], layer_mods[1], layer_mods[2], layer_mods[3]
            xs = xs + attn(norm1(xs), pos_emb, mask)
            xs = xs + ffn(norm2(xs))
            
        return self.after_norm(xs), mask


# ==========================================
# 3. Flow Matching Decoder (CFM)
# ==========================================
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.linear_2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(t)))


class CausalBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 3, padding=0)
        self.norm = nn.LayerNorm(dim_out)
        self.mish = nn.Mish()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (2, 0))
        h = self.mish(self.norm(self.conv(x * mask).transpose(1, 2)).transpose(1, 2))
        return h * mask


class ResnetBlock1D(nn.Module):
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


class ConditionalDecoder(nn.Module):
    def __init__(self, dim: int = 256, meanflow: bool = False):
        super().__init__()
        self.meanflow = meanflow
        self.time_embeddings = nn.Linear(1, 320)
        self.time_mlp = TimestepEmbedding(320)
        self.mixer = nn.Linear(2560, 1280, bias=False) if meanflow else None

        def _resnet(): return ResnetBlock1D(dim, dim, 1280)
        def _attn(): return Attention(query_dim=dim, heads=8, dim_head=64, bias=False)

        self.down_blocks = nn.ModuleList([nn.ModuleList([_resnet(), _attn(), nn.Conv1d(dim, dim, 3, padding=2)]) for _ in range(4)])
        self.mid_blocks = nn.ModuleList([nn.ModuleList([_resnet(), _attn()]) for _ in range(12)])
        self.up_blocks = nn.ModuleList([nn.ModuleList([ResnetBlock1D(dim * 2, dim, 1280), _attn(), nn.ConvTranspose1d(dim, dim, 4, 2, 1)]) for _ in range(3)])
        self.up_blocks.append(nn.ModuleList([ResnetBlock1D(dim * 2, dim, 1280), _attn(), nn.Conv1d(dim, dim, 3, padding=2)]))
        
        self.final_block = CausalBlock1D(dim, dim)
        self.final_proj = nn.Conv1d(dim, 80, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        r: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t_emb = self.time_mlp(torch.cat([torch.sin(self.time_embeddings(t)), torch.cos(self.time_embeddings(t))], dim=-1))
        if self.meanflow and r is not None and self.mixer is not None:
            r_emb = self.time_mlp(torch.cat([torch.sin(self.time_embeddings(r)), torch.cos(self.time_embeddings(r))], dim=-1))
            t_emb = self.mixer(torch.cat([t_emb, r_emb], dim=1))

        h = torch.cat([x, mu, spks.unsqueeze(-1).expand(-1, -1, x.size(-1)), cond], dim=1)
        
        hiddens, masks = [], [mask]
        for block in self.down_blocks:
            b_mods = cast(nn.ModuleList, block)
            resnet, attn, down = b_mods[0], b_mods[1], b_mods[2]
            m = masks[-1]
            h = resnet(h, m, t_emb)
            h = attn(h.transpose(1, 2), attention_mask=(1.0 - m.transpose(1, 2).bool().float()) * -1e10).transpose(1, 2)
            hiddens.append(h)
            h = down(F.pad(h * m, (2, 0)))[..., :-2] if isinstance(down, nn.Conv1d) else down(h * m)
            masks.append(m[:, :, ::2])

        m = masks[-1]
        for block in self.mid_blocks:
            b_mods = cast(nn.ModuleList, block)
            resnet, attn = b_mods[0], b_mods[1]
            h = resnet(h, m, t_emb)
            h = attn(h.transpose(1, 2), attention_mask=(1.0 - m.transpose(1, 2).bool().float()) * -1e10).transpose(1, 2)

        for block in self.up_blocks:
            b_mods = cast(nn.ModuleList, block)
            resnet, attn, up = b_mods[0], b_mods[1], b_mods[2]
            m, skip = masks.pop(), hiddens.pop()
            h = resnet(torch.cat([h[..., :skip.size(-1)], skip], dim=1), m, t_emb)
            h = attn(h.transpose(1, 2), attention_mask=(1.0 - m.transpose(1, 2).bool().float()) * -1e10).transpose(1, 2)
            h = up(F.pad(h * m, (2, 0)))[..., :-2] if isinstance(up, nn.Conv1d) else up(h * m)

        return self.final_proj(self.final_block(h, masks[0])) * masks[0]


class CFM(nn.Module):
    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.estimator = ConditionalDecoder(meanflow=meanflow)


class S3FlowMatcher(nn.Module):
    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.meanflow = meanflow
        self.input_embedding = nn.Embedding(6561, 512)
        self.spk_embed_affine_layer = nn.Linear(192, 80)
        self.encoder = UpsampleConformerEncoder()
        self.encoder_proj = nn.Linear(512, 80)
        self.decoder = CFM(meanflow=meanflow)

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        tokens_len: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_feat: torch.Tensor,
        spk_xvec: torch.Tensor,
        n_timesteps: int = 10
    ) -> torch.Tensor:
        B = tokens.size(0)
        spk_emb = self.spk_embed_affine_layer(F.normalize(spk_xvec, dim=1))
        
        full_tokens = torch.cat([prompt_tokens, tokens], dim=1)
        mask = torch.arange(full_tokens.size(1), device=tokens.device).unsqueeze(0) < (tokens_len + prompt_tokens.size(1)).unsqueeze(1)
        
        h = self.input_embedding(torch.clamp(full_tokens, max=6560)) * mask.unsqueeze(-1)
        h, h_mask = self.encoder(h, mask)
        
        h = h[:, :-6]
        h_mask = h_mask[:, :-6]
        
        mu = self.encoder_proj(h).transpose(1, 2)
        mel_len1 = prompt_feat.size(2)
        
        cond = torch.zeros([B, 80, mu.size(2)], device=mu.device)
        cond[:, :, :mel_len1] = prompt_feat

        x = torch.randn_like(mu)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=x.device)
        if not self.meanflow:
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        for t, r in zip(t_span[:-1], t_span[1:]):
            r_val = r.view(1) if self.meanflow else None
            dx = self.decoder.estimator(x, h_mask.unsqueeze(1), mu, t.view(1), spk_emb, cond, r_val)
            x = x + (r - t) * dx

        return x[:, :, mel_len1:]


# ==========================================
# 4. Vocoder (HiFTGenerator + F0 Predictor)
# ==========================================
class ConvRNNF0Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.condnet = nn.Sequential(
            *[nn.Sequential(weight_norm(nn.Conv1d(80 if i == 0 else 512, 512, 3, padding=1)), nn.ELU()) for i in range(5)]
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.classifier(self.condnet(x).transpose(1, 2)).squeeze(-1))


class Snake(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class HiFiResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel, 1, d, d)) for d in dilations])
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel, 1, 1, 1)) for _ in dilations])
        self.acts1 = nn.ModuleList([Snake(channels) for _ in dilations])
        self.acts2 = nn.ModuleList([Snake(channels) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, a1, c2, a2 in zip(self.convs1, self.acts1, self.convs2, self.acts2):
            x = x + c2(a2(c1(a1(x))))
        return x


class SourceModuleHnNSF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 1)
        
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        F_mat = torch.zeros((f0.size(0), 9, f0.size(-1)), device=f0.device)
        for i in range(9): 
            F_mat[:, i:i + 1, :] = f0 * (i + 1) / 24000
        
        theta = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        phase = (torch.rand(f0.size(0), 9, 1, device=f0.device) * 2 * np.pi) - np.pi
        phase[:, 0, :] = 0
        
        sine = 0.1 * torch.sin(theta + phase)
        uv = (f0 > 10).float()
        noise = (uv * 0.003 + (1 - uv) * (0.1 / 3)) * torch.randn_like(sine)
        sine = sine * uv + noise
        
        return torch.tanh(self.linear(sine.transpose(1, 2))).transpose(1, 2)


class HiFTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.f0_predictor = ConvRNNF0Predictor()
        self.f0_upsamp = nn.Upsample(scale_factor=120) 
        self.m_source = SourceModuleHnNSF()

        self.conv_pre = weight_norm(nn.Conv1d(80, 512, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(512, 256, 16, 8, 4)),
            weight_norm(nn.ConvTranspose1d(256, 128, 11, 5, 3)),
            weight_norm(nn.ConvTranspose1d(128, 64, 7, 3, 2))
        ])
        
        self.source_downs = nn.ModuleList([
            nn.Conv1d(18, 256, 1, 1), nn.Conv1d(18, 128, 16, 8, 4), nn.Conv1d(18, 64, 80, 40, 20)
        ])
        self.source_resblocks = nn.ModuleList([
            HiFiResBlock(256, 7, [1, 3, 5]), HiFiResBlock(128, 7, [1, 3, 5]), HiFiResBlock(64, 11, [1, 3, 5])
        ])
        
        self.resblocks = nn.ModuleList()
        for ch in [256, 128, 64]:
            for k in [3, 7, 11]:
                self.resblocks.append(HiFiResBlock(ch, k, [1, 3, 5]))

        self.conv_post = weight_norm(nn.Conv1d(64, 18, 7, 1, padding=3))
        self.stft_window = torch.hann_window(16)
        self.pad = nn.ReflectionPad1d((1, 0))

    def decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        window = self.stft_window.to(s.device)
        s_stft = torch.stft(s.squeeze(1), 16, 4, 16, window=window, return_complex=True)
        s_stft = torch.cat([s_stft.real, s_stft.imag], dim=1)

        x = self.conv_pre(x)
        for i in range(3):
            x = self.ups[i](F.leaky_relu(x, negative_slope=0.1))
            if i == 2: 
                x = self.pad(x)
            
            x = x + self.source_resblocks[i](self.source_downs[i](s_stft))
            xs = (self.resblocks[i * 3](x) + self.resblocks[i * 3 + 1](x) + self.resblocks[i * 3 + 2](x)) / 3.0
            x = xs

        x = self.conv_post(F.leaky_relu(x, negative_slope=0.1))
        mag = torch.exp(x[:, :9, :])
        phase = torch.sin(x[:, 9:, :])
        
        mag = torch.clip(mag, max=1e2)
        complex_spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        
        wav = torch.istft(complex_spec, 16, 4, 16, window=window)
        return torch.clamp(wav, -0.99, 0.99)


# ==========================================
# 5. Main S3Gen Wrapper
# ==========================================
class S3Gen(nn.Module):
    trim_fade: torch.Tensor

    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.speaker_encoder = CAMPPlus()
        self.flow = S3FlowMatcher(meanflow=meanflow)
        self.mel2wav = HiFTGenerator()

        fade = (torch.cos(torch.linspace(torch.pi, 0, 480)) + 1) / 2
        self.register_buffer("trim_fade", torch.cat([torch.zeros(480), fade]), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.flow.parameters()).dtype

    def embed_ref(self, ref_wav: torch.Tensor, ref_sr: int) -> Dict[str, torch.Tensor]:
        if ref_sr != 24000:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 24000)
        
        ref_wav_16 = torchaudio.functional.resample(ref_wav, 24000, 16000).to(self.dtype)
        
        spk_emb = self.speaker_encoder.inference(ref_wav_16)
        prompt_mels = mel_spectrogram(ref_wav).transpose(1, 2)
        prompt_tokens, prompt_len = self.tokenizer(ref_wav_16.float())
        
        if prompt_mels.size(2) != 2 * prompt_tokens.size(1):
            prompt_tokens = prompt_tokens[:, :prompt_mels.size(2) // 2]
            prompt_len[0] = prompt_tokens.size(1)

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_feat": prompt_mels,
            "spk_xvec": spk_emb
        }

    @torch.inference_mode()
    def generate(
        self,
        speech_tokens: torch.Tensor,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: int = 24000,
        ref_dict: Optional[Dict[str, torch.Tensor]] = None,
        n_cfm_timesteps: int = 10,
        skip_vocoder: bool = False
    ) -> torch.Tensor:
        
        if ref_dict is None and ref_wav is None:
            raise ValueError("Provide exactly one of ref_wav or ref_dict")
            
        ref = ref_dict if ref_dict else self.embed_ref(ref_wav, ref_sr) # type: ignore
        speech_tokens = torch.atleast_2d(speech_tokens).to(self.device)
        token_lens = torch.tensor([st.size(-1) for st in speech_tokens], device=self.device)

        mels = self.flow(
            tokens=speech_tokens, 
            tokens_len=token_lens,
            prompt_tokens=ref["prompt_tokens"].to(self.device),
            prompt_feat=ref["prompt_feat"].to(self.dtype).to(self.device),
            spk_xvec=ref["spk_xvec"].to(self.dtype).to(self.device),
            n_timesteps=2 if self.flow.meanflow else n_cfm_timesteps
        )

        if skip_vocoder: 
            return mels

        f0 = self.mel2wav.f0_predictor(mels)
        s = self.mel2wav.m_source(self.mel2wav.f0_upsamp(f0.unsqueeze(1)).transpose(1, 2)).transpose(1, 2)
        audio = self.mel2wav.decode(mels, s)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)

        fade_buf = cast(torch.Tensor, self.trim_fade)
        trim_len = min(audio.size(-1), fade_buf.size(0))
        audio[:, :, :trim_len] *= fade_buf[:trim_len]
        
        return audio