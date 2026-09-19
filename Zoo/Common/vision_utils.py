from typing import List

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
