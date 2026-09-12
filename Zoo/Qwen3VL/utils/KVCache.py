import torch


class KVCache:
    """Standalone Key-Value Cache for autoregressive generation."""

    def __init__(self, max_batch_size: int = 1, max_seq_len: int = 8192, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self.seen_tokens: int = 0
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = device
        self.dtype = dtype

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key_states: (bs, num_kv_heads, seq_len, head_dim)
            value_states: (bs, num_kv_heads, seq_len, head_dim)
            layer_idx: index of the transformer layer
        Returns:
            Full concatenated key and value states along the sequence length dimension.
        """
        if len(self.k_cache) <= layer_idx:
            # First forward pass: allocate for this layer
            self.k_cache.append(key_states)
            self.v_cache.append(value_states)
            return key_states, value_states
        else:
            # Append current step states
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value_states], dim=2)
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, seq_len: int = 1):
        self.seen_tokens += seq_len

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def reset(self):
        self.k_cache.clear()
        self.v_cache.clear()
        self.seen_tokens = 0