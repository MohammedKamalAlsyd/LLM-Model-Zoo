import torch
from torch import nn
from Zoo.CLIP.SubModels.CLIPEncoder import CLIPEncoder

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.image_size = 224
        self.patch_size = 32
        
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        
        # Converts 3-channel image to 768-dimensional patches
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        
        # (B, 3, 224, 224) -> (B, 768, 7, 7) -> (B, 768, 49) -> (B, 49, 768)
        patch_embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)

        # Prepend the CLS token
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        
        # Add positions
        return embeddings + self.position_embedding(self.position_ids)

class CLIPVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings()
        self.pre_layrnorm = nn.LayerNorm(768, eps=1e-5)
        
        # Encoder is an nn.ModuleList wrapped in an empty module to match HF key `encoder.layers.x`
        self.encoder = CLIPEncoder(hidden_size=768, intermediate_size=3072, num_heads=12)
        
        self.post_layernorm = nn.LayerNorm(768, eps=1e-5)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        hidden_states = self.encoder(hidden_states, causal_attention_mask=False)

        # Get CLS token (first token) and normalize
        pooled_output = hidden_states[:, 0, :]
        return self.post_layernorm(pooled_output)