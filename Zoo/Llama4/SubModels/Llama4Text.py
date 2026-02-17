import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Llama4TextConfig:
    vocab_size: int = 202048
    hidden_size: int = 5120  # 5120 for 40 heads, 128 head dim
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096 * 32
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 200018
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 500000
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    use_qk_norm: bool = True
    no_rope_layer_interval: int = 4
    attention_chunk_size: int = 8192
    attn_temperature_tuning: float = 4
    floor_scale: int = 8192
    attn_scale: float = 0.1


class Llama4TextExperts(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        """Per-expert parameter container.

        This module stores per-expert projection parameters and a small
        non-linearity. The parameters are intentionally named as
        descriptive weight tensors (not tied to framework-specific layer
        names) so you can map them easily from external weight dictionaries.

        Args:
            config: Llama4TextConfig with expert and dimension settings.
        """
        super().__init__()
        self.config = config

        # Number of local experts and dimensionalities
        self.num_experts = int(config.num_local_experts)
        self.hidden_dim = int(config.hidden_size)
        self.expert_hidden_dim = int(config.intermediate_size)

        # Per-expert combined projection: projects input -> [gate_logits, up_features]
        # Shape: (num_experts, hidden_dim, 2 * expert_hidden_dim)
        self.expert_combined_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, 2 * self.expert_hidden_dim)
        )

        # Per-expert down projection: projects expert space back -> hidden_dim
        # Shape: (num_experts, expert_hidden_dim, hidden_dim)
        self.expert_down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_hidden_dim, self.hidden_dim)
        )

        # Small per-expert activation
        self.activation_fn = nn.SiLU()

        # Initialize weights with a normal distribution similar to many model initializations
        nn.init.normal_(self.expert_combined_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_down_proj, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run all experts on a batch of flattened token embeddings.

        The forward method applies every expert to every token in a
        vectorized fashion. For each expert it:
          1. Projects the input to a combined vector containing gate logits
             and expert-up features.
          2. Splits the combined vector into gate logits and up features.
          3. Applies a non-linearity to the up features and projects them
             back to hidden dimension using the expert-specific down
             projection.
          4. Uses the (token, expert) gate scalar to weight that expert's
             contribution and sums across experts producing the final
             output for each token.

        This implementation is intentionally straightforward and stable
        so it can be used as a reference when mapping pretrained
        parameters from external sources.

        Args:
            inputs: Tensor of shape (T, hidden_dim) where T is number of tokens
                    (flattened batch*sequence) and hidden_dim matches config.

        Returns:
            Tensor of shape (T, hidden_dim) representing the aggregated
            expert outputs for each token.
        """

        # combined: (token_count, num_experts_local, 2*expert_hidden_dim)
        hidden_states = inputs.reshape(self.num_experts, -1, self.hidden_dim)
        combined = torch.bmm(hidden_states, self.expert_combined_proj)

        # Split into gate logits and up features
        gate_logits, up_features = combined.split(self.expert_hidden_dim, dim=-1)

        # gate per token & expert -> reduce to a scalar weight in (token_count, num_experts_local, 1)
        gate_scalar = torch.sigmoid(gate_logits.mean(dim=-1, keepdim=True))

        # Non-linear transform of expert up features: (token_count, num_experts_local, expert_hidden_dim)
        up_activated = self.activation_fn(up_features)

        # Project back to hidden dimension using per-expert down projections
        # expert_down_proj: (num_experts_local, expert_hidden_dim, hidden_dim)
        # out_per_expert: (token_count, num_experts_local, hidden_dim)
        out_per_expert = torch.einsum(
            "tef,efd->ted", up_activated, self.expert_down_proj
        )

        # Weight by gate scalars and sum experts: -> (token_count, hidden_dim)
        weighted = out_per_expert * gate_scalar
        aggregated = weighted.sum(dim=1)

        return aggregated


class Llama4TextMLP(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.config = config

        # Descriptive weights names to ease mapping from external checkpoints
        self.gating_weights = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.input_expansion_weights = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.output_projection_weights = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states):
        """Compute feed-forward output for a batch of token embeddings.

        Args:
            hidden_states: Tensor of shape (..., hidden_size)

        Returns:
            Tensor of shape (..., hidden_size)
        """

        gated = self.activation_fn(self.gating_weights(hidden_states))
        expanded = self.input_expansion_weights(hidden_states)
        hidden_after_gating = gated * expanded
        return self.output_projection_weights(hidden_after_gating)


class Llama4TextMoe(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.config = config

        self.top_k = int(config.num_experts_per_tok)
        self.hidden_dim = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)

        # Expert container (holds per-expert parameters)
        self.expert_container = Llama4TextExperts(config)

        # Routing projection (maps token -> expert logits)
        self.routing_weights = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )

        # A shared feed-forward used as a baseline/residual
        self.shared_feedforward = Llama4TextMLP(config)

    def dense_routing(self, router_weights, hidden_states):
        """
        router_weights: (T, E)   -- already sigmoid applied
        hidden_states:  (T, D)
        """

        token_count, hidden_dim = hidden_states.shape
        num_experts_local = self.num_experts

        # Expand tokens across experts: (token_count, num_experts_local, hidden_dim)
        expanded_tokens = hidden_states.unsqueeze(1).expand(
            token_count, num_experts_local, hidden_dim
        )

        # Weight tokens per-expert
        weights = router_weights.unsqueeze(-1)  # (token_count, num_experts_local, 1)
        routed = (
            expanded_tokens * weights
        )  # (token_count, num_experts_local, hidden_dim)

        # Rearrange to run per-expert processing in batch: (num_experts_local, token_count, hidden_dim) -> (num_experts_local*token_count, hidden_dim)
        routed = (
            routed.permute(1, 0, 2)
            .contiguous()
            .view(num_experts_local * token_count, hidden_dim)
        )

        # Run experts using the expert container
        expert_out_flat = self.expert_container(
            routed
        )  # (num_experts_local*token_count, hidden_dim)

        # Reshape back to (num_experts_local, token_count, hidden_dim) and sum across experts -> (token_count, hidden_dim)
        expert_out = expert_out_flat.view(num_experts_local, token_count, hidden_dim)
        final_output = expert_out.sum(dim=0)
        return final_output

    def sparse_routing(self, router_weights, router_top_indices, hidden_states):
        """
        router_weights:     (T, top_k)  -- already sigmoid applied
        router_top_indices: (T, top_k)
        hidden_states:      (T, D)
        """

        final_output = torch.zeros_like(hidden_states)

        # For each expert, gather the tokens assigned to it and run the per-expert
        # processing only on those tokens (sparse execution).
        for expert_id in range(self.num_experts):
            mask = router_top_indices == expert_id
            if not mask.any():
                continue

            token_indices, k_indices = mask.nonzero(as_tuple=True)

            # Inputs for this expert: (N, D)
            expert_input = hidden_states[token_indices]

            # Run the expert container for this expert instance.
            # The expert container currently holds parameters for all experts
            # and runs vectorized; extract per-expert by calling the module on
            # the inputs (it will internally use the right parameters).
            # To run a single expert we can index into parameters manually
            # for better performance; for clarity we run by constructing the
            # per-expert forward here.

            # Compute combined projection for this expert only
            combined = torch.einsum(
                "nd,ho->no",
                expert_input,
                self.expert_container.expert_combined_proj[expert_id],
            )
            gate_logits, up_features = combined.split(
                self.expert_container.expert_hidden_dim, dim=-1
            )
            gate_scalar = torch.sigmoid(gate_logits.mean(dim=-1, keepdim=True))
            up_activated = self.expert_container.activation_fn(up_features)
            expert_out = torch.einsum(
                "nf,fd->nd",
                up_activated,
                self.expert_container.expert_down_proj[expert_id],
            )

            weights = router_weights[token_indices, k_indices].unsqueeze(-1)
            final_output[token_indices] += expert_out * weights * gate_scalar

        return final_output

    def forward(self, hidden_states, mode="dense"):
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Flatten tokens
        token_count = batch_size * seq_length
        hidden_states = hidden_states.view(
            token_count, self.hidden_dim
        )  # (token_count, hidden_dim)

        # Router logits
        router_logits = self.routing_weights(
            hidden_states
        )  # (token_count, num_experts)

        # Top-k selection
        router_top_value, router_top_indices = torch.topk(
            router_logits, self.top_k, dim=1
        )  # (token_count, top_k)

        if mode == "dense":
            # Dense uses full routing weights
            router_weights = torch.sigmoid(router_logits)  # (token_count, num_experts)
            routed_out = self.dense_routing(router_weights, hidden_states)

        elif mode == "sparse":
            # Sparse uses only top-k weights
            router_weights = torch.sigmoid(router_top_value)  # (token_count, top_k)
            routed_out = self.sparse_routing(
                router_weights, router_top_indices, hidden_states
            )
        else:
            raise ValueError("mode must be 'dense' or 'sparse'")

        # Shared feed-forward baseline
        shared_out = self.shared_feedforward(hidden_states)  # (token_count, hidden_dim)
        final = shared_out + routed_out

        return final.view(batch_size, seq_length, hidden_dim)


if __name__ == "__main__":
    config = Llama4TextConfig()
    moe_layer = Llama4TextMoe(config)

    batch_size = 2
    seq_length = 4
    hidden_dim = config.hidden_size

    dummy_input = torch.randn(batch_size, seq_length, hidden_dim)
    # output_dense = moe_layer(dummy_input, mode="dense")
    output_sparse = moe_layer(dummy_input, mode="sparse")

    # print("Dense MoE output shape:", output_dense.shape)
    print("Sparse MoE output shape:", output_sparse.shape)
