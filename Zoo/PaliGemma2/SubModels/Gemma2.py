from torch import nn
from dataclasses import dataclass
import torch


@dataclass
class Gemma2Config:
    vocab_size : int 
    hidden_size : int
    intermediate_size: int
    num_attention_heads : int
    num_hidden_layers: int
    num_key_value_heads: int
    sliding_window: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: int
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    image_token_id: int
    attention_dropout: float
    
    
class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"