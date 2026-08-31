import torch
from torch import nn
from Zoo.CLIP.SubModels.CLIPEncoder import CLIPEncoder

class CLIPTextEmbeddings(nn.Module):
    position_ids: torch.Tensor 
    
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 512)
        self.position_embedding = nn.Embedding(77, 512)
        self.register_buffer("position_ids", torch.arange(77).expand((1, -1)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(self.position_ids[:, :seq_length])
        
        return inputs_embeds + position_embeddings

class CLIPTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings()
        
        self.encoder = CLIPEncoder(hidden_size=512, intermediate_size=2048, num_heads=8)
        
        self.final_layer_norm = nn.LayerNorm(512, eps=1e-5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)

        hidden_states = self.encoder(hidden_states, causal_attention_mask=True)

        last_hidden_state = self.final_layer_norm(hidden_states)

        # Find the highest token ID index (the EOS token) to use as the sequence summary
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.argmax(dim=-1)
        ]

        return pooled_output