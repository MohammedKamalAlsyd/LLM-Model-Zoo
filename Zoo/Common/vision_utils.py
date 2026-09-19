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