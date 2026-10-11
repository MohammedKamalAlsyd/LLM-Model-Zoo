"""SAM3 Video Tracking Memory Engine matching official SAM2/SAM3 weights."""

from typing import List, Optional, Tuple
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


class TrackerTransformer(nn.Module):
    """Two-way transformer decoder matching checkpoint keys `mask_decoder.transformer.*`."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TrackerTwoWayAttentionBlock(hidden_size=hidden_size, inner_dim=128, num_heads=8)
            for _ in range(2)
        ])
        self.final_attn_token_to_image = TrackerCrossAttention(hidden_size, hidden_size, 128, 8)
        self.layer_norm_final_attn = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor, img_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            tokens, img_tokens = layer(tokens, img_tokens)

        attended_tokens, _ = self.final_attn_token_to_image(tokens, img_tokens, img_tokens)
        tokens = self.layer_norm_final_attn(tokens + attended_tokens)
        return tokens, img_tokens


class TrackerMaskDecoder(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer = TrackerTransformer(hidden_size=hidden_size)

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

    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = image_embeddings.shape[0]
        tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight,
            self.obj_score_token.weight,
        ], dim=0).unsqueeze(0).expand(b, -1, -1)

        _, _, h, w = image_embeddings.shape
        img_tokens = image_embeddings.flatten(2).transpose(1, 2)

        tokens, img_tokens = self.transformer(tokens, img_tokens)

        iou_pred = self.iou_prediction_head(tokens[:, 0])
        mask_token_out = tokens[:, 1:5]
        obj_score = self.pred_obj_score_head(tokens[:, 5])

        img_spatial = img_tokens.transpose(1, 2).reshape(b, self.hidden_size, h, w)
        u1 = self.upscale_conv1(img_spatial)
        u1 = self.upscale_layer_norm(u1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if high_res_features is not None and len(high_res_features) > 1:
            u1 = u1 + self.conv_s1(high_res_features[1])
        u2 = self.upscale_conv2(u1)
        if high_res_features is not None and len(high_res_features) > 0:
            u2 = u2 + self.conv_s0(high_res_features[0])

        mask_logits_list = []
        for i in range(4):
            hyper = self.output_hypernetworks_mlps[i](mask_token_out[:, i])
            pred_m = torch.einsum("bc,bchw->bhw", hyper, u2).unsqueeze(1)
            mask_logits_list.append(pred_m)

        all_masks = torch.cat(mask_logits_list, dim=1)
        best_mask_idx = torch.argmax(iou_pred, dim=-1, keepdim=True)
        idx_expanded = best_mask_idx[:, :, None, None].expand(-1, 1, all_masks.shape[2], all_masks.shape[3])
        selected_mask = all_masks.gather(1, idx_expanded)
        return selected_mask, iou_pred, obj_score


class TrackerMemoryAttentionLayer(nn.Module):
    """Single layer matching keys `memory_attention.layers.*`."""

    def __init__(self, hidden_size: int = 256, mem_dim: int = 64) -> None:
        super().__init__()
        self.self_attn = Sam3Attention(hidden_size, 8)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.cross_attn_image = MemoryCrossAttention(hidden_size, mem_dim, hidden_size, 8)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, 2048)
        self.linear2 = nn.Linear(2048, hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

    def forward(self, curr: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        curr = self.layer_norm1(curr + self.self_attn(curr, curr, curr)[0])
        curr = self.layer_norm2(curr + self.cross_attn_image(curr, mem)[0])
        curr = self.layer_norm3(curr + self.linear2(F.relu(self.linear1(curr))))
        return curr


class TrackerMemoryAttention(nn.Module):
    def __init__(self, hidden_size: int = 256, mem_dim: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TrackerMemoryAttentionLayer(hidden_size=hidden_size, mem_dim=mem_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mem = memory if memory_pos is None else (memory + memory_pos)
        for layer in self.layers:
            curr = layer(curr, mem)
        return self.layer_norm(curr)


class MaskDownsamplerBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
        self.layer_norm = nn.LayerNorm(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.relu(x)


class MaskDownsampler(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        channels = [1, 4, 16, 64, 256]
        self.layers = nn.ModuleList([
            MaskDownsamplerBlock(in_c, out_c)
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])
        self.final_conv = nn.Conv2d(256, hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class MemoryFuserBlock(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=7, padding=3, groups=hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pointwise_conv1 = nn.Linear(hidden_size, hidden_size * 4)
        self.pointwise_conv2 = nn.Linear(hidden_size * 4, hidden_size)
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        res = fused
        fused = self.depthwise_conv(fused)
        fused = self.layer_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        fused_flat = self.pointwise_conv2(F.relu(self.pointwise_conv1(fused.permute(0, 2, 3, 1))))
        return res + (fused_flat * self.scale).permute(0, 3, 1, 2)


class MemoryFuser(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryFuserBlock(hidden_size=hidden_size)
            for _ in range(2)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TrackerMemoryEncoder(nn.Module):
    """Processes historical masks and features into compact 64-dim memory tokens."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.mask_downsampler = MaskDownsampler(hidden_size=hidden_size)
        self.memory_fuser = MemoryFuser(hidden_size=hidden_size)
        self.projection = nn.Conv2d(hidden_size, 64, kernel_size=1)

    def forward(self, pix_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_feat = self.mask_downsampler(mask)

        if mask_feat.shape[-2:] != pix_feat.shape[-2:]:
            mask_feat = F.interpolate(mask_feat, size=pix_feat.shape[-2:], mode="bilinear", align_corners=False)

        fused = self.feature_projection(pix_feat) + mask_feat
        fused = self.memory_fuser(fused)
        mem = self.projection(fused)
        return mem.flatten(2).transpose(1, 2)


class PromptMaskEmbed(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.layer_norm1 = nn.LayerNorm(4)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.layer_norm2 = nn.LayerNorm(16)
        self.conv3 = nn.Conv2d(16, hidden_size, kernel_size=1)


class PromptPositionalEmbedding(nn.Module):
    positional_embedding: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(2, 128))


class TrackerPromptEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.mask_embed = PromptMaskEmbed(hidden_size)
        self.no_mask_embed = nn.Embedding(1, hidden_size)
        self.not_a_point_embed = nn.Embedding(1, hidden_size)
        self.point_embed = nn.Embedding(4, hidden_size)
        self.shared_embedding = PromptPositionalEmbedding()


class Sam3TrackerModel(nn.Module):
    """Top-level video tracker matching `tracker_model.*` checkpoint keys."""

    memory_temporal_positional_encoding: torch.Tensor
    no_memory_embedding: torch.Tensor
    no_memory_positional_encoding: torch.Tensor
    no_object_pointer: torch.Tensor
    occlusion_spatial_embedding_parameter: torch.Tensor

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mask_decoder = TrackerMaskDecoder(hidden_size)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.memory_attention = TrackerMemoryAttention(hidden_size=hidden_size, mem_dim=64, num_layers=4)
        self.memory_encoder = TrackerMemoryEncoder(hidden_size)

        # Prompt encoder
        self.prompt_encoder = TrackerPromptEncoder(hidden_size)
        self.object_pointer_proj = TrackerMLP(hidden_size, hidden_size, hidden_size, num_layers=1)
        self.temporal_positional_encoding_projection_layer = nn.Linear(hidden_size, 64)

        # Learnable parameters matching official weights
        self.memory_temporal_positional_encoding = nn.Parameter(torch.zeros(7, 1, 1, 64))
        self.no_memory_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.no_memory_positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.no_object_pointer = nn.Parameter(torch.zeros(1, hidden_size))
        self.occlusion_spatial_embedding_parameter = nn.Parameter(torch.zeros(1, 64))

        self.shared_image_embedding = PromptPositionalEmbedding()

    def encode_memory(self, pix_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Converts frame features and segmentation mask into 64-dim memory tokens."""
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.dtype == torch.bool:
            mask = mask.float()
        return self.memory_encoder(pix_feat, mask)

    def track_step(
        self,
        curr_feat: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]],
        memory_bank: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tracks the active object into the current frame via temporal memory cross-attention.

        Args:
            curr_feat: Current frame deep spatial features [B, 256, H, W].
            high_res_features: Multi-scale shallow FPN layers.
            memory_bank: List of historical memory tensors [B, L, 64].

        Returns:
            Tuple of predicted mask logits [B, 1, H_out, W_out] and object presence score [B, 1].
        """
        b, c, h, w = curr_feat.shape
        flat_curr = curr_feat.flatten(2).transpose(1, 2)

        if not memory_bank:
            attended = flat_curr + self.no_memory_embedding
        else:
            num_mems = min(len(memory_bank), 7)
            recent_mems = memory_bank[-num_mems:]
            positioned_mems = []
            for i, mem in enumerate(reversed(recent_mems)):
                t_pos = self.memory_temporal_positional_encoding[i].view(1, 1, 64)
                positioned_mems.append(mem + t_pos)
            cat_mem = torch.cat(positioned_mems, dim=1)
            attended = self.memory_attention(flat_curr, cat_mem)

        attended_spatial = attended.transpose(1, 2).reshape(b, c, h, w)
        pred_mask, _, obj_score = self.mask_decoder(attended_spatial, high_res_features=high_res_features)
        return pred_mask, obj_score