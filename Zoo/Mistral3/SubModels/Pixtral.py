# ============================================================================
# Pixtral Vision-Language Model Components
# ============================================================================

from torch import nn
import torch
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PixtralConfig:
    """Configuration class for Pixtral model architecture."""
    head_dim: int = 64
    num_heads: int = 16
    attention_dropout: float = 0.0
    hidden_size: int = 1024
    image_size: int = 1540
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    rope_theta: float = 100000.0



# ============================================================================
# Rotary Position Embeddings (RoPE) - 2D Grid-based
# ============================================================================

class PixtraRotaryEmbedding(nn.Module):
    """
    2D Rotary Position Embedding for image tokens.
    
    Key difference from standard RoPE: Each pixel position gets its own frequency
    based on its 2D location (height, width) in the image grid.
    
    Outputs tensor of shape (batch, height * width, dim) with position embeddings
    where each token gets a positional embedding based on its grid position.
    """
    
    inv_freq: torch.Tensor  # Type hint for `register_buffer`

    def __init__(self, config: PixtralConfig) -> None:
        super().__init__()
        self.config = config
        self.rope_base = config.rope_theta
        
        # Compute and register inverse frequencies
        inv_freq = self.compute_default_rope_parameters()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    # --------
    # Helper Methods
    # --------

    def rotate_half(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions of the input.
        
        Used in RoPE to apply rotation: (x1, x2) -> (-x2, x1)
        """
        x1 = hidden_state[..., : hidden_state.shape[-1] // 2]
        x2 = hidden_state[..., hidden_state.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # --------
    # RoPE Computation
    # --------

    def compute_default_rope_parameters(self) -> torch.Tensor:
        """
        Compute inverse frequencies for 2D grid-based RoPE.
        
        Unlike standard RoPE which uses sequence position, this creates a 2D grid
        where each (height, width) position gets separate frequency components.
        
        Returns:
            Tensor of shape (patches_total, dim) with inverse frequencies
        """
        base = self.rope_base
        dim = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )

        # Create 2D grid of patch positions
        max_patches_per_side = self.config.image_size // self.config.patch_size
        h = torch.arange(max_patches_per_side)
        w = torch.arange(max_patches_per_side)

        # Compute base frequencies
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create separate frequency maps for height and width
        freqs_h = torch.outer(h, freqs[::2]).float()  # shape: (patches, dim/4)
        freqs_w = torch.outer(w, freqs[1::2]).float()  # shape: (patches, dim/4)

        # Combine height and width frequencies into 2D grid
        inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),  # (H, W, dim/4)
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),  # (H, W, dim/4)
            ],
            dim=-1,  # (H, W, dim/2)
        ).reshape(-1, dim // 2)  # (H*W, dim/2)

        # Duplicate to match full dimension
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)  # (H*W, dim)
        return inv_freq
    
    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, 
        sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> tuple:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine part of rotary embedding
            sin: Sine part of rotary embedding
            unsqueeze_dim: Dimension to unsqueeze for broadcasting
                - For [batch, heads, seq_len, head_dim] use unsqueeze_dim=1
                - For [batch, seq_len, heads, head_dim] use unsqueeze_dim=2

        Returns:
            Tuple of (q_rotated, k_rotated)
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    # --------
    # Forward Pass
    # --------

    def forward(self, hidden_state: torch.Tensor, position_ids: torch.Tensor) -> tuple:
        """
        Compute cos and sin components of rotary embeddings.
        
        Args:
            hidden_state: Hidden states for dtype conversion
            position_ids: Position indices to look up in inverse frequencies
            
        Returns:
            Tuple of (cos, sin) embeddings
        """
        freqs = self.inv_freq[position_ids]  # shape: (seq_len, dim)
        emb = freqs
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=hidden_state.dtype), sin.to(dtype=hidden_state.dtype)



# ============================================================================
# Feed-Forward Network (MLP)
# ============================================================================

class PixtralMLP(nn.Module):
    """
    Feed-forward network with Gated Linear Units (GLU) architecture.
    
    Structure: (gate_proj * up_proj) -> activation -> down_proj
    This design allows better expressiveness compared to standard dense layers.
    """

    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Projection layers (no bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Gate mechanism: (gate * up) -> activation -> down
        down_proj = self.down_proj(
            self.act_fn(self.up_proj(hidden_states) * self.gate_proj(hidden_states))
        )
        return down_proj


# ============================================================================
# Layer Normalization (RMS Norm)
# ============================================================================

class PixtralRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More computationally efficient than standard LayerNorm while providing
    similar stabilization benefits during training.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Normalized tensor with same shape
        """
        # Store original dtype and convert to float32 for stability
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Compute variance along last dimension
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        # Normalize and scale back to original dtype
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


