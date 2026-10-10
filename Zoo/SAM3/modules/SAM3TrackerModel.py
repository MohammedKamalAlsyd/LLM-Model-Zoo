"""SAM3 Video Tracking Memory Engine matching official SAM2/SAM3 weights."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.SAM3.modules.SAM3Common import Sam3Attention


class TrackerMLP(nn.Module):
    """Matching keys: `.proj_in`, `.layers.0`, `.proj_out`."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj_in(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.proj_out(x)


class TrackerCrossAttention(nn.Module):
    """Internal cross-attention compressing hidden dimension from 256 to 128."""

    def __init__(self, q_dim: int = 256, k_dim: int = 256, inner_dim: int = 128, num_heads: int = 8) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads

        self.q_proj = nn.Linear(q_dim, inner_dim)
        self.k_proj = nn.Linear(k_dim, inner_dim)
        self.v_proj = nn.Linear(k_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim, q_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, None]:
        b, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, q_len, self.inner_dim)), None


class MemoryCrossAttention(nn.Module):
    """Cross-attention matching 64-dim memory keys with 256-dim queries."""

    def __init__(self, q_dim: int = 256, mem_dim: int = 64, inner_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads

        self.q_proj = nn.Linear(q_dim, inner_dim)
        self.k_proj = nn.Linear(mem_dim, inner_dim)
        self.v_proj = nn.Linear(mem_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim, q_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, None]:
        b, q_len, _ = query.shape
        _, k_len, _ = memory.shape

        q = self.q_proj(query).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, q_len, self.inner_dim)), None


class TrackerTwoWayAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int = 256, inner_dim: int = 128, num_heads: int = 8, mlp_dim: int = 2048) -> None:
        super().__init__()
        self.self_attn = Sam3Attention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.cross_attn_token_to_image = TrackerCrossAttention(hidden_size, hidden_size, inner_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = TrackerMLP(hidden_size, mlp_dim, hidden_size, num_layers=0)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        self.cross_attn_image_to_token = TrackerCrossAttention(hidden_size, hidden_size, inner_dim, num_heads)
        self.layer_norm4 = nn.LayerNorm(hidden_size)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.layer_norm1(queries + self.self_attn(queries, queries, queries)[0])
        queries = self.layer_norm2(queries + self.cross_attn_token_to_image(queries, keys, keys)[0])
        queries = self.layer_norm3(queries + self.mlp(queries))
        keys = self.layer_norm4(keys + self.cross_attn_image_to_token(keys, queries, queries)[0])
        return queries, keys


class TrackerMaskDecoder(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([
            TrackerTwoWayAttentionBlock(hidden_size=hidden_size, inner_dim=128, num_heads=8)
            for _ in range(2)
        ])
        self.transformer.final_attn_token_to_image = TrackerCrossAttention(hidden_size, hidden_size, 128, 8)
        self.transformer.layer_norm_final_attn = nn.LayerNorm(hidden_size)

        self.iou_token = nn.Embedding(1, hidden_size)
        self.mask_tokens = nn.Embedding(4, hidden_size)
        self.obj_score_token = nn.Embedding(1, hidden_size)

        self.iou_prediction_head = TrackerMLP(hidden_size, hidden_size, 4, num_layers=1)
        self.pred_obj_score_head = TrackerMLP(hidden_size, hidden_size, 1, num_layers=1)

        self.output_hypernetworks_mlps = nn.ModuleList([
            TrackerMLP(hidden_size, hidden_size, hidden_size // 8, num_layers=1)
            for _ in range(4)
        ])
        self.conv_s0 = nn.Conv2d(hidden_size, hidden_size // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1)

        self.upscale_conv1 = nn.ConvTranspose2d(hidden_size, hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_layer_norm = nn.LayerNorm(hidden_size // 4)
        self.upscale_conv2 = nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, kernel_size=2, stride=2)


class TrackerMemoryAttention(nn.Module):
    def __init__(self, hidden_size: int = 256, mem_dim: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = Sam3Attention(hidden_size, 8)
            layer.layer_norm1 = nn.LayerNorm(hidden_size)
            layer.cross_attn_image = MemoryCrossAttention(hidden_size, mem_dim, hidden_size, 8)
            layer.layer_norm2 = nn.LayerNorm(hidden_size)
            layer.linear1 = nn.Linear(hidden_size, 2048)
            layer.linear2 = nn.Linear(2048, hidden_size)
            layer.layer_norm3 = nn.LayerNorm(hidden_size)
            self.layers.append(layer)
        self.layer_norm = nn.LayerNorm(hidden_size)


class TrackerMemoryEncoder(nn.Module):
    """Processes historical masks and features into compact 64-dim memory tokens."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.mask_downsampler = nn.Module()

        channels = [1, 4, 16, 64, 256]
        self.mask_downsampler.layers = nn.ModuleList()
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            blk = nn.Module()
            blk.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
            blk.layer_norm = nn.LayerNorm(out_c)
            self.mask_downsampler.layers.append(blk)
        self.mask_downsampler.final_conv = nn.Conv2d(256, hidden_size, kernel_size=1)

        self.memory_fuser = nn.Module()
        self.memory_fuser.layers = nn.ModuleList()
        for _ in range(2):
            fuser = nn.Module()
            fuser.depthwise_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=7, padding=3, groups=hidden_size)
            fuser.layer_norm = nn.LayerNorm(hidden_size)
            fuser.pointwise_conv1 = nn.Linear(hidden_size, hidden_size * 4)
            fuser.pointwise_conv2 = nn.Linear(hidden_size * 4, hidden_size)
            fuser.scale = nn.Parameter(torch.ones(hidden_size))
            self.memory_fuser.layers.append(fuser)

        self.projection = nn.Conv2d(hidden_size, 64, kernel_size=1)


class Sam3TrackerModel(nn.Module):
    """Top-level video tracker matching `tracker_model.*` checkpoint keys."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.mask_decoder = TrackerMaskDecoder(hidden_size)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.memory_attention = TrackerMemoryAttention(hidden_size=hidden_size, mem_dim=64, num_layers=4)
        self.memory_encoder = TrackerMemoryEncoder(hidden_size)

        # Prompt encoder
        self.prompt_encoder = nn.Module()
        self.prompt_encoder.mask_embed = nn.Module()
        self.prompt_encoder.mask_embed.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.prompt_encoder.mask_embed.layer_norm1 = nn.LayerNorm(4)
        self.prompt_encoder.mask_embed.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.prompt_encoder.mask_embed.layer_norm2 = nn.LayerNorm(16)
        self.prompt_encoder.mask_embed.conv3 = nn.Conv2d(16, hidden_size, kernel_size=1)

        self.prompt_encoder.no_mask_embed = nn.Embedding(1, hidden_size)
        self.prompt_encoder.not_a_point_embed = nn.Embedding(1, hidden_size)
        self.prompt_encoder.point_embed = nn.Embedding(4, hidden_size)
        self.prompt_encoder.shared_embedding = nn.Module()
        self.prompt_encoder.shared_embedding.positional_embedding = nn.Parameter(torch.zeros(2, 128))

        self.object_pointer_proj = TrackerMLP(hidden_size, hidden_size, hidden_size, num_layers=1)
        self.temporal_positional_encoding_projection_layer = nn.Linear(hidden_size, 64)

        # Learnable buffers matching official weights
        self.register_parameter("memory_temporal_positional_encoding", nn.Parameter(torch.zeros(7, 1, 1, 64)))
        self.register_parameter("no_memory_embedding", nn.Parameter(torch.zeros(1, 1, hidden_size)))
        self.register_parameter("no_memory_positional_encoding", nn.Parameter(torch.zeros(1, 1, hidden_size)))
        self.register_parameter("no_object_pointer", nn.Parameter(torch.zeros(1, hidden_size)))
        self.register_parameter("occlusion_spatial_embedding_parameter", nn.Parameter(torch.zeros(1, 64)))

        self.shared_image_embedding = nn.Module()
        self.shared_image_embedding.positional_embedding = nn.Parameter(torch.zeros(2, 128))