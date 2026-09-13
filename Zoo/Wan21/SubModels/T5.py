# Copyright 2024-2025 The Alibaba Wan Team Authors and Project Contributors.
import html
import logging
import math
import os
import re
import string
from typing import Callable, List, Optional, Tuple, Union

import ftfy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

__all__ = ["T5Encoder", "T5EncoderModel"]


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):

    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in (2, 3)
            mask_view = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask_view == 0, torch.finfo(x.dtype).min)

        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_buckets = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_buckets)
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
        ).long()
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttentionBlock(nn.Module):

    def __init__(self, dim: int, dim_attn: int, dim_ffn: int, num_heads: int, num_buckets: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=pos_bias))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5Encoder(nn.Module):
    """
    UMT5-XXL Encoder matching official state_dict directly.
    """

    def __init__(
        self,
        vocab_size: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttentionBlock(dim, dim_attn, dim_ffn, num_heads, num_buckets, dropout)
            for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1))
        for block in self.blocks:
            x = block(x, mask=mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5EncoderModel:
    """
    Unified High-level Text Encoder Interface with built-in HuggingFace Tokenizer.
    """

    def __init__(
        self,
        text_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cpu",
        checkpoint_path: Optional[str] = None,
        tokenizer_path: str = "google/umt5-xxl",
        shard_fn: Optional[Callable] = None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = torch.device(device)

        self.model = T5Encoder().to(dtype=dtype, device=self.device).eval().requires_grad_(False)

        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"Loading T5 weights from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, assign=True)

        if shard_fn is not None:
            self.model = shard_fn(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, tokenizer_path="google/umt5-xxl", device="cpu", dtype=torch.bfloat16):
        pth = os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        tok = os.path.join(checkpoint_dir, "google/umt5-xxl") if os.path.exists(os.path.join(checkpoint_dir, "google/umt5-xxl")) else tokenizer_path
        return cls(checkpoint_path=pth, tokenizer_path=tok, device=device, dtype=dtype)

    def __call__(self, texts: List[str], device: Optional[torch.device] = None) -> List[torch.Tensor]:
        dev = device or self.device
        cleaned_texts = [self._clean(t) for t in texts]
        tokens = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
        )
        ids = tokens.input_ids.to(dev)
        mask = tokens.attention_mask.to(dev)
        seq_lens = mask.gt(0).sum(dim=1).long()

        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def _clean(self, text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()