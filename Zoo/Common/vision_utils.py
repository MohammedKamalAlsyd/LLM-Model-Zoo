from typing import List, Tuple

import torch
import torch.nn.functional as F

def interpolate_2d_pos_embed(
    pos_embed: torch.Tensor,
    target_height: int,
    target_width: int,
    patch_size: int,
) -> torch.Tensor:
    """Bicubically interpolates a 2D patch position embedding table.

    Args:
        pos_embed: (num_positions, dim) or (1, num_positions, dim)
        target_height: Input image height in pixels
        target_width: Input image width in pixels
        patch_size: Patch size in pixels (e.g., 14)
    """
    if pos_embed.ndim == 2:
        pos_embed = pos_embed.unsqueeze(0)

    _, num_pos, dim = pos_embed.shape
    new_h = target_height // patch_size
    new_w = target_width // patch_size

    if num_pos == new_h * new_w:
        return pos_embed

    grid = int(num_pos ** 0.5)
    pos_embed = pos_embed.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )
    return pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)


def generate_block_attention_mask(
    patch_counts: List[int],
    device: torch.device,
    as_boolean: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Constructs a block-diagonal attention mask for variable-sized image sequences.

    Ensures patches from Image A attend exclusively to Image A, preventing cross-image leakage.

    Args:
        patch_counts: List containing the number of patches per image.
        device: Target execution device.
        as_boolean: If True, returns a boolean mask (True=attend, False=mask) for SDPA.
                    If False, returns an additive float mask (0.0=attend, -inf=mask).
        dtype: Floating point precision used when as_boolean=False.

    Returns:
        Tensor of shape (1, 1, total_tokens, total_tokens).
    """
    total_tokens = sum(patch_counts)

    if as_boolean:
        # True = allow attention, False = disallow attention (SDPA native convention)
        mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
        start = 0
        for count in patch_counts:
            end = start + count
            mask[start:end, start:end] = True
            start = end
    else:
        neg_inf = torch.finfo(dtype).min
        mask = torch.full((total_tokens, total_tokens), fill_value=neg_inf, dtype=dtype, device=device)
        start = 0
        for count in patch_counts:
            end = start + count
            mask[start:end, start:end] = 0.0
            start = end

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


def get_vision_cu_seqlens(grid_thw: torch.Tensor, merge_temporal: bool = False) -> torch.Tensor:
    """Calculates cumulative sequence lengths for packed variable-length attention.

    Args:
        grid_thw: Tensor of shape (num_images, 3) holding [T, H, W] grid sizes.
        merge_temporal: Whether to group temporal frames together.

    Returns:
        Tensor of shape (num_chunks + 1,) containing segment boundaries.
    """
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int = 2) -> torch.Tensor:
    """Calculates (H, W) 2D coordinates for axial RoPE in spatial-merge block major ordering.

    Args:
        grid_thw: Tensor of shape (num_items, 3) with [T, H, W] per visual input.
        spatial_merge_size: Spatial downsampling window size (default 2).

    Returns:
        Tensor of shape (total_tokens, 2) with (h, w) coordinates.
    """
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block_shape = (h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        coords = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        position_ids.append(coords)
    return torch.cat(position_ids, dim=0)


def _interpolation_axis_taps_weights(
    index: torch.Tensor, size: torch.Tensor, side: int, align_corners: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates 1D bilinear interpolation taps and distance weights."""
    index = index.to(torch.float32)
    if align_corners:
        src = index * (side - 1) / torch.clamp(size - 1, min=1)
    else:
        src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    weights = (1.0 - distance).clamp(min=0.0)
    return taps, weights


def get_vision_interpolation_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int = 2,
    align_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes 2D bilinear interpolation taps and weights for learned position embeddings.

    Args:
        grid_thw: Tensor of shape (num_items, 3) holding [T, H, W] per item.
        num_grid_per_side: Dimension of pre-trained position embedding table.
        spatial_merge_size: Downsampling factor (default 2).
        align_corners: Resampling coordinate convention.

    Returns:
        indices: Tensor of shape (total_tokens, 4) with 2D corner indices.
        weights: Tensor of shape (total_tokens, 4) with interpolation weights.
    """
    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device

    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)

    total_elements = int(counts.sum().item())
    within = (torch.arange(total_elements, device=device) - starts) % (heights * widths)

    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    h_taps, h_weights = _interpolation_axis_taps_weights(row, heights, side, align_corners=align_corners)
    w_taps, w_weights = _interpolation_axis_taps_weights(col, widths, side, align_corners=align_corners)

    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def pack_latents_2d(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Packs 2D spatial latents into 2x2 flattened patch token sequences.

    Used by FLUX and MMDiT backbones:
    [B, C, H, W] -> [B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size]

    Args:
        latents: Spatial tensor of shape (batch_size, channels, height, width).
        patch_size: Spatial patch aggregation window (default: 2).

    Returns:
        Tensor of shape (batch_size, num_patches, channels * patch_size^2).
    """
    b, c, h, w = latents.shape
    p = patch_size
    latents = latents.view(b, c, h // p, p, w // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // p) * (w // p), c * p * p)


def unpack_latents_2d(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    patch_size: int = 2,
) -> torch.Tensor:
    """Unpacks flattened patch tokens back into a 2D spatial feature map.

    [B, (H // patch_size) * (W // patch_size), C * patch_size^2] -> [B, C, H, W]

    Args:
        latents: Packed patch tokens of shape (batch_size, num_patches, packed_dim).
        height: Target pixel image height.
        width: Target pixel image width.
        vae_scale_factor: Spatial compression ratio of the VAE (default: 8).
        patch_size: Patch de-aggregation window (default: 2).

    Returns:
        Tensor of shape (batch_size, channels, latent_height, latent_width).
    """
    b, _, packed_channels = latents.shape
    p = patch_size
    latent_h = 2 * (int(height) // (vae_scale_factor * 2))
    latent_w = 2 * (int(width) // (vae_scale_factor * 2))
    channels = packed_channels // (p * p)

    latents = latents.view(b, latent_h // p, latent_w // p, channels, p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, channels, latent_h, latent_w)


def prepare_multiaxis_coordinate_grid(
    height_patches: int,
    width_patches: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generates 3D spatial-temporal positional coordinates for DiT RoPE: [H * W, 3].

    Column 0: Time/Batch axis (zeros for static images).
    Column 1: Height index coordinates.
    Column 2: Width index coordinates.

    Args:
        height_patches: Number of vertical patch tokens.
        width_patches: Number of horizontal patch tokens.
        device: Target execution device.
        dtype: Output coordinate data type.

    Returns:
        Tensor of shape (height_patches * width_patches, 3).
    """
    grid = torch.zeros(height_patches, width_patches, 3, device=device, dtype=dtype)
    grid[..., 1] += torch.arange(height_patches, device=device, dtype=dtype)[:, None]
    grid[..., 2] += torch.arange(width_patches, device=device, dtype=dtype)[None, :]
    return grid.reshape(height_patches * width_patches, 3)