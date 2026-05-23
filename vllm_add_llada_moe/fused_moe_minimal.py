"""
最小化 FusedMoE 实现
- 复制 vLLM 的 fused_experts_impl 核心逻辑
- Buffer 在 Python 层分配（方便后续外部控制）
- 去除量化、chunking 等复杂特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from vllm import _custom_ops as ops
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        moe_align_block_size,
        invoke_fused_moe_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.triton_utils import tl


def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-K 专家选择
    
    Returns:
        topk_weights: [M, topk]
        topk_ids: [M, topk]
    """
    M, num_experts = gating_output.shape
    
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=gating_output.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)
    token_expert_indicies = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)
    
    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights, topk_ids


def fused_experts_impl_minimal(
    tensors: dict,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
):
    """
    精简版 fused_experts 实现（从 vLLM 提取）
    
    关键特性：
    - 从 tensors 字典直接使用预分配的 buffer（形状已在计算图中定义）
    - 调用 vLLM 的 Triton kernel
    - 无量化、无 chunking
    - 原地写入输出（moe_output）
    
    Args:
        tensors: 张量字典，包含：
            - x_normed_2: [M, K] - 输入
            - moe_cache1: [M, topk, N*2] - 第一次 GEMM 输出
            - moe_cache2: [M*topk, N] - silu_and_mul 输出
            - moe_cache3: [M, topk, K] - 第二次 GEMM 输出
            - moe_output: [M, K] - 最终输出
        w1: [E, N*2, K] - gate + up projection 合并
        w2: [E, K, N] - down projection
        topk_weights: [M, topk]
        topk_ids: [M, topk]
        activation: 激活函数 ("silu" or "gelu")
    """
    # 从 tensors 读取
    hidden_states = tensors['x_normed_2']
    moe_cache1 = tensors['moe_cache1']
    moe_cache2 = tensors['moe_cache2']
    moe_cache3 = tensors['moe_cache3']  # 在 reuse_plan 中会与 cache1 复用同一内存
    moe_output = tensors['moe_output']
    # 检查约束
    assert hidden_states.size(1) == w1.size(2), f"Hidden size mismatch"
    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    topk = topk_ids.size(1)
    
    # 获取 Triton kernel 配置
    config = try_get_optimal_moe_config(
        w1_shape=w1.size(),
        w2_shape=w2.size(),
        top_k=topk,
        dtype=None,  # 使用默认
        M=M,
    )
    
    # =====================================================================
    # 直接使用 tensors 中预分配的 buffers（形状已经在计算图中定义好）
    # =====================================================================
    # moe_cache1: [M, topk, N*2] - 第一次 GEMM 输出
    # moe_cache2: [M*topk, N] - silu_and_mul 输出 (保持 2D 用于后续 GEMM)
    # moe_cache3: [M, topk, K] - 第二次 GEMM 输出（与 cache1 复用内存）
    intermediate_cache1 = moe_cache1  # 已经是 [M, topk, N]
    intermediate_cache2 = moe_cache2  # 已经是 [M*topk, N//2]
    intermediate_cache3 = moe_cache3  # 已经是 [M, topk, K]
    
    # Triton compute type
    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {hidden_states.dtype}")
    
    # =====================================================================
    # Step 1: Token 排序和对齐（为了高效 GEMM）
    # =====================================================================
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        config['BLOCK_SIZE_M'],
        E,  # global_num_experts
        expert_map=None,
    )
    
    # =====================================================================
    # Step 2: 第一次 GEMM (hidden @ w1 -> intermediate_cache1)
    # =====================================================================
    invoke_fused_moe_kernel(
        A=hidden_states,
        B=w1,
        C=intermediate_cache1,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,  # 第一次不乘权重
        top_k=topk,
        config=config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    )
    
    # =====================================================================
    # Step 3: 激活函数 (SiLU * gate_up 或 GELU)
    # =====================================================================
    if activation == "silu":
        # SiLU(gate) * up
        torch.ops._C.silu_and_mul(
            intermediate_cache2,
            intermediate_cache1.view(-1, N)
        )
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(
            intermediate_cache2,
            intermediate_cache1.view(-1, N)
        )
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # =====================================================================
    # Step 4: 第二次 GEMM (intermediate_cache2 @ w2 -> intermediate_cache3)
    # =====================================================================
    invoke_fused_moe_kernel(
        A=intermediate_cache2,
        B=w2,
        C=intermediate_cache3,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,  # 第二次乘权重
        top_k=1,  # 注意这里是 1（因为 cache2 已经是展开的）
        config=config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    )
    
    # =====================================================================
    # Step 5: 聚合结果（按 token 求和，原地写入 moe_output）
    # =====================================================================
    ops.moe_sum(
        intermediate_cache3.view(*intermediate_cache3.size()),
        moe_output
    )


# 别名（向后兼容）
fused_experts = fused_experts_impl_minimal


# ============================================================================
# 便捷包装 - 给 LLaDAMoESparseMoeBlock 使用
# ============================================================================

class SimpleFusedMoE(nn.Module):
    """
    简化的 FusedMoE 模块
    - Buffer 在 fused_experts_impl_minimal 中分配
    - 后续可以修改为外部传入
    - 支持从 HF checkpoint 加载权重
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
        renormalize: bool = True,
        **kwargs
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        
        # 专家权重：合并格式
        # w13_weight[:, :intermediate_size, :] = gate_proj (w1)
        # w13_weight[:, intermediate_size:, :] = up_proj (w3)
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                dtype=params_dtype
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=params_dtype
            )
        )
        
        # 初始化（实际使用时从 checkpoint 加载）
        nn.init.xavier_uniform_(self.w13_weight)
        nn.init.xavier_uniform_(self.w2_weight)
        
        # 注册 weight_loader 到参数上（vLLM 的 DefaultLoader 会调用）
        self.w13_weight.weight_loader = self.w13_weight_loader
        self.w2_weight.weight_loader = self.w2_weight_loader
    
    def w13_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """
        w13_weight 的权重加载器（gate_proj 和 up_proj）
        """
        # 解析 weight_name: "experts.{expert_id}.{gate_proj|up_proj}.weight"
        parts = weight_name.split(".")
        for i, part in enumerate(parts):
            if part == "experts" and i + 1 < len(parts):
                expert_id = int(parts[i + 1])
                if "gate_proj" in weight_name:
                    # 加载到前半部分
                    param.data[expert_id, :self.intermediate_size, :] = loaded_weight
                elif "up_proj" in weight_name:
                    # 加载到后半部分
                    param.data[expert_id, self.intermediate_size:, :] = loaded_weight
                return
        
        # 如果无法解析，报错
        raise ValueError(f"Cannot parse weight_name for w13_weight: {weight_name}")
    
    def w2_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """
        w2_weight 的权重加载器（down_proj）
        """
        # 解析 weight_name: "experts.{expert_id}.down_proj.weight"
        parts = weight_name.split(".")
        for i, part in enumerate(parts):
            if part == "experts" and i + 1 < len(parts):
                expert_id = int(parts[i + 1])
                if "down_proj" in weight_name:
                    param.data[expert_id, :, :] = loaded_weight
                return
        
        # 如果无法解析，报错
        raise ValueError(f"Cannot parse weight_name for w2_weight: {weight_name}")
    
    def forward(
        self,
        tensors: dict,
        router_logits: torch.Tensor,
    ):
        """
        Args:
            tensors: 张量字典，包含 x_normed_2, moe_cache1/2/3, moe_output
            router_logits: [M, num_experts]
        """
        # Top-K 选择
        topk_weights, topk_ids = fused_topk(
            router_logits,
            self.top_k,
            renormalize=self.renormalize,
        )
        
        # 调用 fused_experts_impl，直接传入 tensors
        # cache1 和 cache3 会通过 reuse_plan 自动复用同一块内存
        fused_experts_impl_minimal(
            tensors=tensors,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
        )
