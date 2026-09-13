import torch
import torch.nn as nn
import torch.nn.functional as F
from Zoo.SAM3.SubModels.SAM3Common import Sam3Attention

# ============================================================================
# MLPs & Downscaled Cross-Attention
# ============================================================================

class TrackerMLP(nn.Module):
    """Matches keys: .proj_in, .layers.0, .proj_out"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj_in(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.proj_out(x)


class TrackerCrossAttention(nn.Module):
    """
    Downscaled cross-attention where internal dimension is 128 instead of 256.
    q_proj: 256 -> 128
    k_proj: 256 -> 128
    v_proj: 256 -> 128
    o_proj: 128 -> 256
    """
    def __init__(self, q_dim: int = 256, k_dim: int = 256, inner_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads

        self.q_proj = nn.Linear(q_dim, inner_dim)
        self.k_proj = nn.Linear(k_dim, inner_dim)
        self.v_proj = nn.Linear(k_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim, q_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        b, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.inner_dim)
        return self.o_proj(out), None


class MemoryCrossAttention(nn.Module):
    """
    Cross attention to memory where Keys and Values have dim 64 and Queries have dim 256.
    q_proj: 256 -> 256
    k_proj: 64  -> 256
    v_proj: 64  -> 256
    o_proj: 256 -> 256
    """
    def __init__(self, q_dim: int = 256, mem_dim: int = 64, inner_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads

        self.q_proj = nn.Linear(q_dim, inner_dim)
        self.k_proj = nn.Linear(mem_dim, inner_dim)
        self.v_proj = nn.Linear(mem_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim, q_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, None]:
        b, q_len, _ = query.shape
        _, k_len, _ = memory.shape

        q = self.q_proj(query).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.inner_dim)
        return self.o_proj(out), None


# ============================================================================
# Tracker Mask Decoder
# ============================================================================

class TrackerTwoWayAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int = 256, inner_dim: int = 128, num_heads: int = 8, mlp_dim: int = 2048):
        super().__init__()
        self.self_attn = Sam3Attention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.cross_attn_token_to_image = TrackerCrossAttention(q_dim=hidden_size, k_dim=hidden_size, inner_dim=inner_dim, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = TrackerMLP(hidden_size, mlp_dim, hidden_size, num_layers=0)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        self.cross_attn_image_to_token = TrackerCrossAttention(q_dim=hidden_size, k_dim=hidden_size, inner_dim=inner_dim, num_heads=num_heads)
        self.layer_norm4 = nn.LayerNorm(hidden_size)

    def forward(self, queries, keys):
        q = queries + self.self_attn(queries, queries, queries)[0]
        queries = self.layer_norm1(q)

        q = queries + self.cross_attn_token_to_image(queries, keys, keys)[0]
        queries = self.layer_norm2(q)

        q = queries + self.mlp(queries)
        queries = self.layer_norm3(q)

        k = keys + self.cross_attn_image_to_token(keys, queries, queries)[0]
        keys = self.layer_norm4(k)
        return queries, keys


class TrackerTwoWayTransformer(nn.Module):
    def __init__(self, hidden_size: int = 256, inner_dim: int = 128, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            TrackerTwoWayAttentionBlock(hidden_size=hidden_size, inner_dim=inner_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.final_attn_token_to_image = TrackerCrossAttention(q_dim=hidden_size, k_dim=hidden_size, inner_dim=inner_dim, num_heads=num_heads)
        self.layer_norm_final_attn = nn.LayerNorm(hidden_size)

    def forward(self, queries, keys):
        for layer in self.layers:
            queries, keys = layer(queries, keys)
        q = queries + self.final_attn_token_to_image(queries, keys, keys)[0]
        return self.layer_norm_final_attn(q), keys


class TrackerMaskDecoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.transformer = TrackerTwoWayTransformer(hidden_size=hidden_size, inner_dim=128, num_layers=2)

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


# ============================================================================
# Tracker Memory Attention
# ============================================================================

class MemoryAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int = 256, mem_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.self_attn = Sam3Attention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.cross_attn_image = MemoryCrossAttention(q_dim=hidden_size, mem_dim=mem_dim, inner_dim=hidden_size, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.linear1 = nn.Linear(hidden_size, 2048)
        self.linear2 = nn.Linear(2048, hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, memory):
        h = x + self.self_attn(x, x, x)[0]
        x = self.layer_norm1(h)

        h = x + self.cross_attn_image(x, memory)[0]
        x = self.layer_norm2(h)

        h = x + self.linear2(F.relu(self.linear1(x)))
        return self.layer_norm3(h)


class TrackerMemoryAttention(nn.Module):
    def __init__(self, hidden_size: int = 256, mem_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(hidden_size=hidden_size, mem_dim=mem_dim) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)


# ============================================================================
# Tracker Memory Encoder & Prompt Encoder
# ============================================================================

class MaskDownsamplerLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
        self.layer_norm = nn.LayerNorm(out_c)


class TrackerMemoryFuserLayer(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Linear(dim, dim * 4)
        self.pointwise_conv2 = nn.Linear(dim * 4, dim)
        self.scale = nn.Parameter(torch.ones(dim))


class TrackerMemoryEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

        # Exact checkpoint channels: 1 -> 4 -> 16 -> 64 -> 256
        self.mask_downsampler = nn.Module()
        self.mask_downsampler.layers = nn.ModuleList([
            MaskDownsamplerLayer(1, 4),
            MaskDownsamplerLayer(4, 16),
            MaskDownsamplerLayer(16, 64),
            MaskDownsamplerLayer(64, 256),
        ])
        self.mask_downsampler.final_conv = nn.Conv2d(256, hidden_size, kernel_size=1)

        self.memory_fuser = nn.Module()
        self.memory_fuser.layers = nn.ModuleList([
            TrackerMemoryFuserLayer(hidden_size) for _ in range(2)
        ])
        self.projection = nn.Conv2d(hidden_size, 64, kernel_size=1)


class TrackerPromptEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.mask_embed = nn.Module()
        # Exact checkpoint channels: 1 -> 4 -> 16 -> 256
        self.mask_embed.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.mask_embed.layer_norm1 = nn.LayerNorm(4)
        self.mask_embed.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.mask_embed.layer_norm2 = nn.LayerNorm(16)
        self.mask_embed.conv3 = nn.Conv2d(16, hidden_size, kernel_size=1)

        self.no_mask_embed = nn.Embedding(1, hidden_size)
        self.not_a_point_embed = nn.Embedding(1, hidden_size)
        self.point_embed = nn.Embedding(4, hidden_size)

        self.shared_embedding = nn.Module()
        self.shared_embedding.positional_embedding = nn.Parameter(torch.zeros(2, 128))


# ============================================================================
# Full Tracker Model
# ============================================================================

class Sam3TrackerModel(nn.Module):
    """
    Submodule matching all `tracker_model.*` checkpoint keys and tensor dimensions.
    """
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.mask_decoder = TrackerMaskDecoder(hidden_size)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.memory_attention = TrackerMemoryAttention(hidden_size=hidden_size, mem_dim=64, num_layers=4)
        self.memory_encoder = TrackerMemoryEncoder(hidden_size)
        self.prompt_encoder = TrackerPromptEncoder(hidden_size)

        self.object_pointer_proj = TrackerMLP(hidden_size, hidden_size, hidden_size, num_layers=1)
        self.temporal_positional_encoding_projection_layer = nn.Linear(hidden_size, 64)

        # Standalone parameter buffers with exact checkpoint shapes
        self.register_parameter("memory_temporal_positional_encoding", nn.Parameter(torch.zeros(7, 1, 1, 64)))
        self.register_parameter("no_memory_embedding", nn.Parameter(torch.zeros(1, 1, hidden_size)))
        self.register_parameter("no_memory_positional_encoding", nn.Parameter(torch.zeros(1, 1, hidden_size)))
        self.register_parameter("no_object_pointer", nn.Parameter(torch.zeros(1, hidden_size)))
        self.register_parameter("occlusion_spatial_embedding_parameter", nn.Parameter(torch.zeros(1, 64)))

        self.shared_image_embedding = nn.Module()
        self.shared_image_embedding.positional_embedding = nn.Parameter(torch.zeros(2, 128))