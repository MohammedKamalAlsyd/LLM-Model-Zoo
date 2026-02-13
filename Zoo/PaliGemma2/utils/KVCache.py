"""Key-Value Cache Module for Efficient Inference.

This module provides a caching mechanism for storing and retrieving key and value
tensors during the decoding phase of transformer models. KV-caching is a critical
optimization technique that reduces computation and memory access by avoiding
redundant attention computations on previously processed tokens.

Classes:
    KVCache: Manages the storage and concatenation of key-value pairs across layers.
"""

from typing import List, Tuple

import torch

__all__ = ["KVCache"]


class KVCache:
    """Key-Value cache for efficient transformer inference.

    Caches key and value tensors from multi-head attention layers to avoid
    recomputing attention for previously processed tokens during generation.
    This is essential for efficient autoregressive decoding in large language models.

    The cache stores separate lists of keys and values, indexed by layer number,
    allowing incremental updates as new tokens are generated.

    Attributes:
        key_cache (List[torch.Tensor]): List of cached key tensors, one per layer.
            Each tensor has shape (batch_size, num_heads, seq_length, head_dim).
        value_cache (List[torch.Tensor]): List of cached value tensors, one per layer.
            Each tensor has shape (batch_size, num_heads, seq_length, head_dim).
    """

    def __init__(self) -> None:
        """Initialize an empty KV cache.

        Creates empty lists for storing key and value tensors from each layer
        of the model. The cache is populated lazily as layers process tokens.
        """
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value tensors and return concatenated cache.

        On the first call for a given layer, initializes the cache with the
        provided tensors. On subsequent calls, concatenates new tensors to the
        existing cache along the sequence length dimension (dim=-2).

        This operation enables efficient incremental decoding where only the
        current token's attention needs to be computed, with cached values
        used for previous tokens.

        Args:
            new_key (torch.Tensor): New key tensor to cache, shape
                (batch_size, num_heads, 1, head_dim) for single token, or
                (batch_size, num_heads, new_seq_len, head_dim) for multiple tokens.
            new_value (torch.Tensor): New value tensor to cache, shape
                (batch_size, num_heads, 1, head_dim) for single token, or
                (batch_size, num_heads, new_seq_len, head_dim) for multiple tokens.
            layer_idx (int): Index of the transformer layer (0-indexed).
                Must be non-negative.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Concatenated key cache of shape (batch_size, num_heads, total_seq_len, head_dim)
                - Concatenated value cache of shape (batch_size, num_heads, total_seq_len, head_dim)

        Raises:
            ValueError: If layer_idx is negative.
            RuntimeError: If tensor shapes are incompatible or inconsistent
                between key and value tensors.

        Shape:
            - Input key: (B, num_heads, seq_len, head_dim)
            - Input value: (B, num_heads, seq_len, head_dim)
            - Output: (B, num_heads, total_cached_seq_len, head_dim)

        Notes:
            - First call for a layer: initializes cache with new_key and new_value
            - Subsequent calls: concatenate along sequence dimension (dim=-2)
            - Key and value shapes must match except for sequence length dimension
        """
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        if new_key.shape != new_value.shape:
            raise RuntimeError(
                f"key and value shapes must match. Got key: {new_key.shape}, "
                f"value: {new_value.shape}"
            )

        if not self.key_cache:
            # First call: initialize cache for this layer
            self.key_cache.append(new_key)
            self.value_cache.append(new_value)
        else:
            # Subsequent calls: concatenate with existing cache
            if layer_idx >= len(self.key_cache):
                raise RuntimeError(
                    f"layer_idx {layer_idx} out of range for cache with "
                    f"{len(self.key_cache)} layers"
                )

            # Verify shape compatibility (all dims except seq_len must match)
            cached_key = self.key_cache[layer_idx]
            if (
                new_key.shape[0] != cached_key.shape[0]
                or new_key.shape[1] != cached_key.shape[1]
                or new_key.shape[3] != cached_key.shape[3]
            ):
                raise RuntimeError(
                    f"Incompatible shapes: new key shape {new_key.shape} does not match "
                    f"cached key shape {cached_key.shape} in batch, heads, or head_dim"
                )

            # Concatenate along sequence dimension (dim=-2)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], new_key], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], new_value], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
