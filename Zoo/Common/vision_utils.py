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


def replace_image_tokens(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_features: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Universally injects visual features into placeholder token slots.

    Works seamlessly across PaliGemma 2, Ministral 3, LLaVA, and Pixtral.
    Handles arbitrary batch sizes, variable image resolutions, and ensures
    exact dtype and device alignment before scattering.

    Args:
        input_ids: Tensor of token IDs of shape (batch_size, seq_len).
        inputs_embeds: Full sequence token embeddings of shape (batch_size, seq_len, hidden_size).
        image_features: Projected vision features of shape (total_tokens, hidden_size)
                        or (batch_size, num_patches, hidden_size).
        image_token_id: Integer ID representing the image placeholder token (e.g., 10 or 257152).

    Returns:
        Tensor with vision embeddings scattered into image token positions,
        matching the shape and dtype of `inputs_embeds`.

    Raises:
        ValueError: If the number of placeholder tokens in `input_ids` does not
                    match the total number of visual tokens in `image_features`.
    """
    # 1. Locate placeholder positions
    image_mask = (input_ids == image_token_id)
    num_placeholders = int(image_mask.sum().item())

    # 2. Flatten all leading dimensions of image_features (supports 2D or 3D)
    hidden_size = inputs_embeds.shape[-1]
    flat_features = image_features.reshape(-1, hidden_size)
    num_features = flat_features.shape[0]

    # 3. Strict token count validation
    if num_placeholders != num_features:
        raise ValueError(
            f"Multimodal token count mismatch! "
            f"Found {num_placeholders} placeholder tokens in 'input_ids' (ID={image_token_id}), "
            f"but received {num_features} visual features from vision projector."
        )

    # 4. Expand mask to full embedding dimensionality (batch_size, seq_len, hidden_size)
    expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)

    # 5. Type and device safety alignment
    features_aligned = flat_features.to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    ).contiguous()

    # 6. Scatter into embedding tensor
    return inputs_embeds.masked_scatter(expanded_mask, features_aligned)