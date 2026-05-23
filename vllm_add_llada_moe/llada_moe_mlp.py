"""
LLaDA MoE MLP and Router implementation for vLLM.
Adapted from modeling_lladamoe.py
Updated to use vLLM FusedMoE for better performance.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用精简版 FusedMoE（buffer 在 Python 层管理）
from vllm_add_llada_moe.fused_moe_minimal import SimpleFusedMoE as FusedMoE


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
        # This provides significant performance improvements through:
        # - Token permutation and batching
        # - Fused Triton kernels
        # - Efficient memory access patterns
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

    def forward(self, tensors: dict):
        """
        Args:
            tensors: 张量字典
        """
        # 从 tensors 读取输入和输出
        x_normed_2 = tensors['x_normed_2']
        moe_output = tensors['moe_output']
        
        # 获取 chunk 策略（由 model 设置）
        strategy = tensors.get('_chunk_strategy')
        use_chunkwise = strategy and strategy.get('chunk_mlp', False)
        
        if use_chunkwise:
            # Chunk-wise MoE 计算
            N = x_normed_2.shape[0]
            num_chunks = strategy.get('num_chunks_mlp', 5)  # 从策略读取
            tile_size = (N + num_chunks - 1) // num_chunks
            
            # 先一次性计算完整的 router_logits (很小，[N, 64])
            router_logits_full = self.gate(x_normed_2)
            if self.expert_bias is not None:
                router_logits_full = router_logits_full + self.expert_bias
            
            for i in range(0, N, tile_size):
                end_i = min(i + tile_size, N)
                t_i = end_i - i  # 当前 chunk 的实际大小
                
                # 创建当前 chunk 的 tensors（使用切片，都是 view，不占额外显存）
                chunk_tensors = {
                    'x_normed_2': x_normed_2[i:end_i],  # 输入切片 [t_i, hidden]
                    'moe_cache1': tensors['moe_cache1'][0:t_i],  # cache 切片（复用 buffer 的前 t_i 部分）
                    'moe_cache2': tensors['moe_cache2'][0:t_i * self.top_k],  # cache2 是 2D
                    'moe_cache3': tensors['moe_cache3'][0:t_i],  # cache 切片
                    'moe_output': moe_output[i:end_i],  # 输出切片 [t_i, hidden]
                }
                
                # 使用预先计算好的 router_logits 切片
                router_logits_chunk = router_logits_full[i:end_i]
                
                # FusedMoE 计算（使用 chunk_tensors）
                self.experts(chunk_tensors, router_logits_chunk)
        
        else:
            # 非 chunk：一次性计算
            # Router logits: [N, num_experts]
            router_logits = self.gate(x_normed_2)
            
            # Add expert bias if configured
            if self.expert_bias is not None:
                router_logits = router_logits + self.expert_bias
            
            # FusedMoE 计算（使用 tensors 中预分配的 moe_cache1/2/3 和 moe_output）
            # 原地写入 tensors['moe_output']，不返回值
            self.experts(tensors, router_logits)


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
    
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # One-hot encode selected experts
    expert_mask = F.one_hot(selected_experts, num_experts)
    
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

