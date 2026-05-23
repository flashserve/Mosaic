"""
LLaDA MoE MLP using vLLM FusedMoE for better performance.
This replaces the original llada_moe_mlp.py implementation.
"""

from typing import Optional
import torch
import torch.nn as nn

from vllm_add_llada_moe.fused_moe import FusedMoE


class LLaDAMoEMLP(nn.Module):
    """
    Single MLP module used in MoE experts, dense layers, and shared experts.
    Supports three types: 'dense', 'expert', 'shared_expert'
    """
    def __init__(self, config, mlp_type: str):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Set intermediate_size based on mlp_type
        if mlp_type == 'dense':
            self.intermediate_size = config.dense_intermediate_size
        elif mlp_type == 'expert':
            self.intermediate_size = config.expert_intermediate_size
        elif mlp_type == 'shared_expert':
            self.intermediate_size = config.shared_expert_intermediate_size
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")
        
        # SwiGLU architecture: gate_proj, up_proj, down_proj
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function (SiLU for LLaDA-MoE)
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SwiGLU(x) = down_proj(act(gate_proj(x)) * up_proj(x))
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDAMoESparseMoeBlock(nn.Module):
    """
    Sparse Mixture of Experts block using vLLM's FusedMoE.
    Much faster than the original implementation with expert loop.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = False
        
        # Router: linear layer that computes expert scores
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Use vLLM's FusedMoE instead of manual expert loop
        # This provides significant performance improvements
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            params_dtype=torch.bfloat16,
            reduce_results=False,  # We handle reduction ourselves
            renormalize=self.norm_topk_prob,
            activation="silu",  # Match LLaDA-MoE activation
        )
        
        self.score_func = config.moe_router_score_function
        
        # Optional expert bias
        if config.moe_router_enable_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        else:
            self.expert_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [total_tokens, hidden_size]
        Returns:
            output: same shape as hidden_states
        """
        # Flatten if needed: [B, T, C] -> [B*T, C]
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
        else:
            # Already flattened [N, C]
            batch_size = 1
            sequence_length = hidden_states.size(0)
            hidden_dim = hidden_states.size(1)
        
        # Router logits: [batch * sequence_length, num_experts]
        router_logits = self.gate(hidden_states)
        
        # Add expert bias if configured (before passing to FusedMoE)
        if self.expert_bias is not None:
            router_logits = router_logits + self.expert_bias
        
        # FusedMoE handles:
        # 1. Top-k expert selection
        # 2. Token routing and permutation
        # 3. Expert computation (fused GEMM)
        # 4. Result aggregation with routing weights
        final_hidden_states = self.experts(hidden_states, router_logits)
        
        # Reshape back to original shape
        if len(original_shape) == 3:
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states


def load_balancing_loss_func(
    gate_logits: torch.Tensor, 
    num_experts: int, 
    top_k: int = 2, 
    attention_mask: Optional[torch.Tensor] = None
) -> float:
    """
    Computes auxiliary load balancing loss for MoE.
    From Switch Transformer (https://arxiv.org/abs/2101.03961).
    
    Args:
        gate_logits: Tuple of tensors [batch_size * seq_len, num_experts] from each layer
        num_experts: Number of experts
        top_k: Number of experts selected per token
        attention_mask: Optional mask (not used for diffusion models)
    
    Returns:
        Load balancing loss value
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )
    
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # One-hot encode selected experts
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    
    if attention_mask is None:
        # Compute percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        
        # Compute average routing probability per expert
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # Handle attention mask (for non-diffusion models)
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )
        
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )
    
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

