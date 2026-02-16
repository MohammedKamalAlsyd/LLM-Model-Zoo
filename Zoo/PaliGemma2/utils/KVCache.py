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
        
    def num_items(self) -> int:
        """Return the number of cached items (tokens) in the cache.
        Assumes all layers have the same sequence length in their cached tensors.
        Returns:
            int: The number of cached tokens, determined by the sequence length
                dimension of the first layer's cached key tensor. Returns 0 if
                the cache is empty.
        """
        if not self.key_cache:
            return 0
        return self.key_cache[0].shape[2]  # seq_length dimension

    def update(
            self,
            new_key: torch.Tensor,
            new_value: torch.Tensor,
            layer_idx: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Update cache with new key-value tensors."""
            
            # If we haven't visited this layer yet, append new entries
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(new_key)
                self.value_cache.append(new_value)
            else:
                # We are revisiting this layer (generation step), concatenate along seq_len dim
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], new_key], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], new_value], dim=-2
                )

            return self.key_cache[layer_idx], self.value_cache[layer_idx]