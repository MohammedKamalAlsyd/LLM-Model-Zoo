import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3, bias=in_proj_bias) # 3 for q, k, v
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=out_proj_bias)
        self.hidden_size_per_head = hidden_size // n_heads
        self.n_heads = n_heads
        
    def forward(self, x, causal_mask):
        batch_size, seq_length, hidden_size = x.shape
        
        # Apply linear projection to get q, k, v
        q, k, v = self.in_proj(x).chunk(3, dim=-1)  # Each is (batch_size, seq_length, hidden_size)

        # Reshape: (B, Seq, H, Dim) -> Transpose to (B, H, Seq, Dim)
        q = q.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute scaled dot-product attention
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,       # Set this to None if using is_causal
            is_causal=causal_mask # Pass the boolean flag here
            # scale=None          # Remove this arg. Default is correct (1 / sqrt(dim))
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        return self.out_proj(output)  # (batch_size, seq_length, hidden_size)
    
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias= in_proj_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias= in_proj_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias= in_proj_bias)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias= out_proj_bias)
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads

    def forward(self, x, context):
        batch_size, seq_length, hidden_size = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        return self.out_proj(output)  # (batch_size, seq_length, hidden_size)


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_layer(x)

        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + residual
    
    
class VAE_AttentionBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.attention = SelfAttention(1, in_channels)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        residual = x
        
        
        x = self.group_norm(x)
        x = x.view(batch_size, channels, height * width).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
        x = x.transpose(1, 2)  # (B, H*W, C) -> (B, C, H*W)
        x = self.attention(x, causal_mask=False)  # (B, H*W, C)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)  # (B, H*W, C) -> (B, C, H, W)
        return x + residual