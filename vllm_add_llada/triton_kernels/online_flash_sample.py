# file: online_flash_sample.py
"""
包含一个基于 Online Softmax 算法的 Triton Kernel。
此算法理论上是单循环，但其串行执行的特性导致在GPU上性能较差。
主要用于教学和性能对比。
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _online_kernel(
    # --- 指针 ---
    logits_ptr,
    output_indices_ptr,
    output_probs_ptr,
    # --- 维度 ---
    M, # seq_len
    N, # vocab_size
    # --- 步长 ---
    stride_m,
    stride_n,
):
    """
    Online Softmax 的 Triton Kernel 实现。
    """
    # 每个程序实例处理一行
    pid_m = tl.program_id(axis=0)
    row_start_ptr = logits_ptr + pid_m * stride_m

    # 初始化 Online Softmax 的三个核心变量
    max_val = -float("inf")  # 当前最大值
    argmax_idx = 0           # 当前最大值的索引
    denominator = 0.0        # 动态缩放的累加和
    
    # 真正的 one-pass 遍历
    # 注意：这是一个串行循环，性能会很差，因为它逐个加载元素
    for k in range(0, N):
        val = tl.load(row_start_ptr + k * stride_n).to(tl.float32)
        
        if val > max_val:
            denominator = denominator * tl.exp(max_val - val) + 1.0
            max_val = val
            argmax_idx = k
        else:
            denominator += tl.exp(val - max_val)

    max_prob = 1.0 / denominator
    tl.store(output_indices_ptr + pid_m, argmax_idx)
    tl.store(output_probs_ptr + pid_m, max_prob)


def fused_argmax_softmax_online(logits: torch.Tensor):
    """
    启动 Online Softmax Triton Kernel 的封装函数。
    """
    assert logits.dim() == 2 and logits.is_cuda, "输入必须是2D的CUDA张量"
    seq_len, vocab_size = logits.shape
    
    output_indices = torch.empty(seq_len, dtype=torch.int64, device=logits.device)
    output_probs = torch.empty(seq_len, dtype=torch.float32, device=logits.device)
    
    grid = (seq_len, )

    # 这个内核是串行的，不需要BLOCK_SIZE_N，但API需要它
    _online_kernel[grid](
        logits, output_indices, output_probs,
        seq_len, vocab_size, logits.stride(0), logits.stride(1)
    )
    return output_indices, output_probs