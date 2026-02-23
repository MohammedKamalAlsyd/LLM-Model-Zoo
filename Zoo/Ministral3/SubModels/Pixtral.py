# ============================================================================
# Pixtral Vision-Language Model Components
# ============================================================================

from torch import nn
import torch
from dataclasses import dataclass
from typing import Optional
from utils.rotate_functions import apply_rotary_pos_emb


# ============================================================================
# Configuration
# ============================================================================


class PixtralConfig:
    """Configuration class for Pixtral model architecture (Vision Tower)."""

    head_dim: int = 64
    attention_dropout: float = 0.0
    head_dim: int = 64
    hidden_size: int = 1024
    image_size: int = 1540
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    rope_theta: float = 100000.0
    num_channels: int = 3


# --------
# Helper Methods
# --------


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    """
    Generate position IDs for a list of patch embeddings based on their 2D grid positions.
    Args:
        patch_embeds_list: List of patch embedding tensors, each of shape (batch, channels, height, width)
        max_width: Maximum width of the image in terms of patches (image_size // patch_size)
    Returns:
        Tensor of shape (total_patches,) containing position IDs for each patch based on its grid position.
    """
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


def generate_block_attention_mask(patch_embeds_list, tensor):
    """
    Generate a block-diagonal attention mask for a sequence of patch embeddings.
    This mask allows full attention within each image's patches but prevents attention across different images in the batch.
    Args:
        patch_embeds_list: List of patch embedding tensors, each of shape (batch, channels, height, width)
        tensor: The input tensor for which the attention mask is being generated (used to determine dtype and device)
    Returns:
        A block-diagonal attention mask tensor of shape (batch, 1, seq_len, seq_len) where seq_len is the total number of patches across all images in the batch. The mask has 0s for positions within the same image and -inf for positions across different images, allowing attention only within each image's patches.
    """
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device
    )

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask


# ============================================================================
# Rotary Position Embeddings (RoPE) - 2D Grid-based
# ============================================================================


class PixtralRotaryEmbedding(nn.Module):
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
        ).reshape(
            -1, dim // 2
        )  # (H*W, dim/2)

        # Duplicate to match full dimension
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)  # (H*W, dim)
        return inv_freq

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
        hidden_states = hidden_states.to(dtype=torch.float32)

        # Compute variance along last dimension
        variance = hidden_states.pow(2).mean(
            -1, keepdim=True
        )  # shape: (batch, seq_len, 1)

        # Normalize and scale back to original dtype
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# Layer Normalization (RMS Norm)
# ============================================================================


class PixtralAttention(nn.Module):
    """
    Multi-head attention mechanism for Pixtral model.

    Key features:
    - Supports both self-attention and cross-attention
    - Uses 2D RoPE for positional embeddings
    - Configurable number of heads and head dimensions
    """

    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # Ensure hidden size is divisible by number of heads
        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), "Hidden size must be divisible by number of heads"

        # Projection layers for query, key, value (no bias)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Output projection layer (no bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Dropout for attention probabilities
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention output given query, key, value tensors.

        Args:
            query: Query tensor of shape (batch, heads, seq_len_q, head_dim)
            key: Key tensor of shape (batch, heads, seq_len_kv, head_dim)
            value: Value tensor of shape (batch, heads, seq_len_kv, head_dim)
            attention_mask: Optional mask tensor for attention (broadcastable to (batch, heads, seq_len_q, seq_len_kv))
            scaling: Scaling factor for dot product attention (e.g., 1/sqrt(head_dim))
        Returns:
            Attention output tensor of shape (batch, heads, seq_len_q, head_dim)
        """
        # Compute scaled dot product attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Compute attention probabilities
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute multi-head attention output.
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask tensor for attention (broadcastable to (batch, heads, seq_len, seq_len))
            position_embeddings: Optional tuple of (cos, sin) tensors for RoPE
            output_attentions: Whether to return attention weights
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size, patches, _ = hidden_states.size()

        # Project hidden states to query, key, value
        query = (
            self.q_proj(hidden_states)
            .view(batch_size, patches, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, seq_len, head_dim)
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, patches, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, seq_len, head_dim)
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, patches, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, seq_len, head_dim)

        # Apply RoPE if position embeddings are provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Compute attention output
        scaling_factor = 1.0 / (self.head_dim**0.5)
        attn_output, attn_weights = self.attention_forward(
            query, key, value, attention_mask=attention_mask, scaling=scaling_factor
        )

        # Project back to hidden size
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, patches, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


# ============================================================================
# Attention Layer with Residual Connection
# ============================================================================
class PixtralAttentionLayer(nn.Module):
    """
    Single layer of attention followed by feed-forward network with residual connections.
    """

    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size)
        self.feed_forward = PixtralMLP(config)
        self.attention = PixtralAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`): Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# ============================================================================
# Transformer consisting of multiple attention layers
# ============================================================================
class PixtralTransformer(nn.Module):
    """
    Stack of attention layers for the Pixtral model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(PixtralAttentionLayer(config))

    def forward(
        self,
        input_embeds,
        attention_mask,
        position_embeddings,
        output_attentions,
        output_hidden_states,
        return_dict=False,
    ):
        """
        Args:
            input_embeds: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask tensor (broadcastable to (batch, heads, seq_len, seq_len))
            position_embeddings: Tuple of (cos, sin) tensors for RoPE
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dictionary of outputs or a tuple
        """
        hidden_states = input_embeds
        all_attentions = (
            []
        )  # intialize it to avoid type error when output_attentions is False
        encoder_states = (
            []
        )  # intialize it to avoid type error when output_hidden_states is False

        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + [hidden_states]

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + [layer_outputs[1]]

        if output_hidden_states:
            encoder_states = encoder_states + [hidden_states]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )

        outputs = {
            "last_hidden_state": hidden_states,
            "hidden_states": tuple(encoder_states) if output_hidden_states else None,
            "attentions": tuple(all_attentions) if output_attentions else None,
        }

        return outputs


# ============================================================================
# Vision Model for Pixtral
# ============================================================================


class PixtralVisionModel(nn.Module):
    """
    Vision model for Pixtral architecture.

    This model processes image inputs and produces hidden states that can be
    fed into a language model for vision-language tasks.
    """

    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.patch_size = config.patch_size
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ):
        """
        Args:
            pixel_values: Tensor of shape (batch, channels, height, width) containing the input images.
            image_sizes: Optional tensor of shape (batch, 2) containing the original height and width of each image in the batch. If None, it is assumed all images have the same size as the input tensor.
            output_hidden_states: Whether to return hidden states from all layers of the transformer.
            output_attentions: Whether to return attention weights from all layers of the transformer.
        """
        if image_sizes is None:
            batch_size, _, height, width = pixel_values.shape
            image_sizes = torch.tensor([(height, width)] * batch_size)

        # pass images through initial convolution independently
        target_dtype = self.patch_conv.weight.dtype
        patch_embeds = self.patch_conv(pixel_values.to(dtype=target_dtype))
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(1).T for p in patch_embeds_list], dim=0
        ).unsqueeze(0)
        patch_embeds = self.ln_pre(
            patch_embeds
        )  # Shape: (1, total_patches (Sum of images patches), hidden_size)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        )  # Shape: (total_patches,)

        position_embeddings = self.patch_positional_embedding(
            patch_embeds, position_ids
        )  # Tuple of (cos, sin) tensors for RoPE, each of shape (total_patches, hidden_size)

        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )

        return self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )  # Note Rope Added in each Transformer Layer not single time here
