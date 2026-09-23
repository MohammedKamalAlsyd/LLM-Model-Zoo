"""Universal Multimodal Sequence and Coordinate Utilities for VLMs.

Handles modality-aware coordinate generation, 3D Multimodal Rotary Position
Embeddings (M-RoPE) scheduling, embedding scatter replacement, and cross-modal
alignment for PaliGemma 2, Ministral-3, and Qwen3-VL backbones.
"""

import itertools
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


# ============================================================================
# Token Embedding Injection & Replacement
# ============================================================================

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


# ============================================================================
# 3D Multimodal Rotary Position (M-RoPE) Geometry
# ============================================================================

def get_vision_grid_3d_positions(
    start_pos: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Constructs 3D [T, H, W] spatial-temporal coordinate grids for visual tokens.

    Args:
        start_pos: Base scalar position offset where the visual chunk begins.
        grid_thw: Tensor or 1D array of 3 elements [Time, Height, Width] representing
            the visual patch dimensions before spatial downsampling.
        spatial_merge_size: Spatial patch reduction factor (default: 2 for 2x2 pooling).
        device: Device to allocate position coordinate tensors on.

    Returns:
        Tensor of shape (3, num_tokens) holding [T, H, W] coordinates for every patch.
    """
    if device is None:
        device = grid_thw.device if isinstance(grid_thw, torch.Tensor) else torch.device("cpu")

    sms = int(spatial_merge_size)
    t = int(grid_thw[0].item()) if isinstance(grid_thw, torch.Tensor) else int(grid_thw[0])
    h = (int(grid_thw[1].item()) if isinstance(grid_thw, torch.Tensor) else int(grid_thw[1])) // sms
    w = (int(grid_thw[2].item()) if isinstance(grid_thw, torch.Tensor) else int(grid_thw[2])) // sms

    pos_t = torch.zeros(t, device=device, dtype=torch.long)
    pos_h = torch.arange(h, device=device, dtype=torch.long) + start_pos
    pos_w = torch.arange(w, device=device, dtype=torch.long) + start_pos

    t_grid, h_grid, w_grid = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
    
    # Shape: (3, t * h * w)
    grid_coords = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1) + start_pos
    return grid_coords


def build_3d_position_ids(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates 3D M-RoPE coordinates across interleaved text and vision sequences.

    Assigns:
        - Text Tokens: T = H = W = position (degenerates gracefully to standard 1D RoPE).
        - Vision Tokens: T = time/frame index, H = row index, W = column index.

    Also derives `rope_deltas`, which bridges sequence token length with 2D/3D coordinate
    expansion to support single-step autoregressive decoding.

    Args:
        input_ids: LongTensor of shape (batch_size, seq_len).
        mm_token_type_ids: Binary LongTensor of shape (batch_size, seq_len) where
            0 represents Text tokens and 1 represents Multimodal tokens.
        image_grid_thw: Optional Tensor of shape (num_images, 3) holding [T, H, W]
            for each image/video input in the batch.
        spatial_merge_size: Patch merger downsampling factor (default: 2).

    Returns:
        pos_ids: LongTensor of shape (3, batch_size, seq_len) with [T, H, W] positions.
        deltas: LongTensor of shape (batch_size, 1) containing rope delta offsets for KV caching.
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device
    sms = int(spatial_merge_size)

    pos_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)
    rope_deltas: List[int] = []

    img_iter = iter(image_grid_thw) if image_grid_thw is not None else None

    for b in range(bs):
        token_types = mm_token_type_ids[b].tolist()
        # Group contiguous chunks of text (0) and vision (1) tokens
        groups = [(modality, len(list(group))) for modality, group in itertools.groupby(token_types)]
        cur_pos: int = 0
        b_positions: List[torch.Tensor] = []

        for modality, length in groups:
            if modality == 0:
                # -----------------------------------------------------------
                # Text Segment: T = H = W = cur_pos (1D equivalence)
                # -----------------------------------------------------------
                text_pos = torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + cur_pos
                b_positions.append(text_pos)
                cur_pos += length

            else:
                # -----------------------------------------------------------
                # Visual Segment: 3D Grid [T, H, W]
                # -----------------------------------------------------------
                if img_iter is None:
                    raise ValueError(
                        "image_grid_thw must be provided when mm_token_type_ids indicates visual tokens."
                    )
                grid = next(img_iter)
                v_pos = get_vision_grid_3d_positions(
                    start_pos=cur_pos,
                    grid_thw=grid,
                    spatial_merge_size=sms,
                    device=device,
                )
                b_positions.append(v_pos)

                # Advance position along the maximum physical dimension of the 2D/3D patch
                step = int(max(int(grid[1].item()), int(grid[2].item()))) // sms
                cur_pos += step

        # Concatenate segments along sequence dimension
        all_pos = torch.cat(b_positions, dim=1)  # (3, seq_len)
        pos_ids[:, b] = all_pos

        # Delta between maximum spatial index reached and logical 1D sequence length
        rope_deltas.append(int(all_pos.max().item()) + 1 - seq_len)

    deltas = torch.tensor(rope_deltas, device=device).unsqueeze(1)
    return pos_ids, deltas


# ============================================================================
# Autoregressive Decode Step Coordinate Helper
# ============================================================================

def make_decode_3d_position_ids(
    batch_size: int,
    seq_len: int,
    past_length: int,
    device: torch.device,
    rope_deltas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Constructs 3D M-RoPE coordinates for single-token incremental decode steps.

    Args:
        batch_size: Batch size dimension.
        seq_len: Current sequence step length (typically 1 during decode).
        past_length: Total number of previously processed tokens stored in KV cache.
        device: Device to place position coordinates on.
        rope_deltas: Optional Tensor of shape (batch_size, 1) computed during the prefill phase.

    Returns:
        Tensor of shape (3, batch_size, seq_len) aligned with 3D coordinate space.
    """
    pos = torch.arange(
        past_length, past_length + seq_len, device=device, dtype=torch.long
    ).view(1, 1, -1).expand(3, batch_size, -1)

    if rope_deltas is not None:
        pos = pos + rope_deltas.unsqueeze(0)

    return pos