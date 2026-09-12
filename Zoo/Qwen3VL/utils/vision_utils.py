import torch
import torch.nn.functional as F


def get_vision_cu_seqlens(grid_thw: torch.Tensor, merge_temporal: bool = False) -> torch.Tensor:
    """Calculates cumulative sequence lengths for packed variable-length attention."""
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int = 2) -> torch.Tensor:
    """
    Computes (H, W) 2D coordinates for axial rotary position embeddings,
    re-ordered into spatial-merge block major ordering.
    Returns: (total_tokens, 2)
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    weights = (1 - distance).clamp(min=0)
    return taps, weights


def get_vision_interpolation_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int = 2,
    align_corners: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes bilinear interpolation taps (indices) and weights for learned 2D
    position embedding tables of size (num_grid_per_side, num_grid_per_side).
    """
    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device

    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    
    # Cast total count to int to satisfy torch.arange typing overload
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