"""Universal Key-Value Cache for autoregressive decoding."""

from typing import List, Tuple
import torch


class KVCache:
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        """Returns sequence length of cached tokens."""
        return self.key_cache[0].shape[-2] if self.key_cache else 0

    def update(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new key/value states along the sequence dimension."""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(new_key)
            self.value_cache.append(new_value)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], new_key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_value], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self) -> None:
        """Clears the cache for a new prompt."""
        self.key_cache.clear()
        self.value_cache.clear()