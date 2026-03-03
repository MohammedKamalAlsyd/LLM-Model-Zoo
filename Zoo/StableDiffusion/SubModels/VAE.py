import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(
            hidden_size, hidden_size * 3, bias=in_proj_bias
        )  # 3 for q, k, v
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=out_proj_bias)
        self.hidden_size_per_head = hidden_size // n_heads
        self.n_heads = n_heads

    def forward(self, x, causal_mask):
        batch_size, seq_length, hidden_size = x.shape

        # Apply linear projection to get q, k, v
        q, k, v = self.in_proj(x).chunk(
            3, dim=-1
        )  # Each is (batch_size, seq_length, hidden_size)

        # Reshape: (B, Seq, H, Dim) -> Transpose to (B, H, Seq, Dim)
        q = q.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)

        # Compute scaled dot-product attention
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # Set this to None if using is_causal
            is_causal=causal_mask,  # Pass the boolean flag here
            # scale=None          # Remove this arg. Default is correct (1 / sqrt(dim))
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, hidden_size)
        )

        return self.out_proj(output)  # (batch_size, seq_length, hidden_size)


class CrossAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=in_proj_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=in_proj_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=in_proj_bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=out_proj_bias)
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
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, hidden_size)
        )

        return self.out_proj(output)  # (batch_size, seq_length, hidden_size)


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

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
        x = x.view(batch_size, channels, height * width).transpose(
            1, 2
        )  # (B, C, H*W) -> (B, H*W, C)
        x = x.transpose(1, 2)  # (B, H*W, C) -> (B, C, H*W)
        x = self.attention(x, causal_mask=False)  # (B, H*W, C)
        x = x.transpose(1, 2).view(
            batch_size, channels, height, width
        )  # (B, H*W, C) -> (B, C, H, W)
        return x + residual


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(
                3, 128, kernel_size=3, padding=1
            ),  # (Batch_Size, 3, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2, padding=0
            ),  # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=2, padding=0
            ),  # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(
                512, 512, kernel_size=3, stride=2, padding=0
            ),  # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):  # type: ignore
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(
                    x, (0, 1, 0, 1), mode="constant", value=0
                )  # Pad right and bottom by 1 pixel
            x = module(x)

        mean, log_var = x.chunk(
            2, dim=1
        )  # (Batch_Size, 8, Height/8, Width/8) -> Two tensors of shape (Batch_Size, 4, Height/8, Width/8)
        log_var = torch.clamp(
            log_var, min=-30.0, max=20.0
        )  # Clamp log_var to prevent numerical issues
        std = torch.exp(log_var / 2)  # (Batch_Size, 4, Height/8, Width/8)

        x = mean + std * noise  # Reparameterization trick
        x *= 0.18215  # Scale factor used in Stable Diffusion
        return x


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):  # type: ignore
        # x: (Batch_Size, 4, Height / 8, Width / 8)

        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x
