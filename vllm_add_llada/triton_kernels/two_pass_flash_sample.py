# file: two_pass_flash_sample.py
"""
包含一个高效的、基于并行化思想的 Two-Pass Triton Kernel。
这个算子通过两次在SRAM中的扫描，高效地计算 argmax 和对应的 softmax 概率值。
这是在 GPU 上实现此类任务的推荐方法。
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _two_pass_kernel(
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
    # --- 元参数 ---
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    高效的 Two-Pass Triton Kernel。
    """
    # 每个程序实例处理一行
    pid_m = tl.program_id(axis=0)
    row_start_ptr = logits_ptr + pid_m * stride_m

    # -----------------------------------------------------------
    # 第一遍扫描 (在SRAM中): 找出当前行的最大值(max_val)和其索引(argmax_idx)
    # -----------------------------------------------------------
    max_val = -float("inf")
    argmax_idx = 0
    
    # 以 BLOCK_SIZE_N 为单位，遍历所有列
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        col_offsets = k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = col_offsets < N
        block_logits = tl.load(row_start_ptr + col_offsets * stride_n, mask=mask, other=-float("inf"))
        
        block_max_val = tl.max(block_logits, axis=0)
        
        if block_max_val > max_val:
            max_val = block_max_val
            is_max_mask = (block_logits == max_val)
            indices_in_block = tl.where(is_max_mask, col_offsets, -1)
            argmax_idx = tl.max(indices_in_block, axis=0)

    # -----------------------------------------------------------
    # 第二遍扫描 (在SRAM中): 使用第一遍找到的 max_val 来计算 softmax 的分母
    # -----------------------------------------------------------
    denominator = 0.0
    
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        col_offsets = k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = col_offsets < N
        block_logits = tl.load(row_start_ptr + col_offsets * stride_n, mask=mask, other=0.0)
        
        stable_logits = block_logits - max_val
        block_exps = tl.exp(stable_logits.to(tl.float32))
        
        denominator += tl.sum(block_exps, axis=0)

    # -----------------------------------------------------------
    # 最终计算与存储
    # -----------------------------------------------------------
    max_prob = 1.0 / denominator
    tl.store(output_indices_ptr + pid_m, argmax_idx)
    tl.store(output_probs_ptr + pid_m, max_prob)


def fused_argmax_softmax_twopass(logits: torch.Tensor):
    """
    启动 Two-Pass Triton Kernel 的封装函数。
    """
    assert logits.dim() == 2 and logits.is_cuda, "输入必须是2D的CUDA张量"
    
    seq_len, vocab_size = logits.shape
    
    output_indices = torch.empty(seq_len, dtype=torch.int64, device=logits.device)
    output_probs = torch.empty(seq_len, dtype=torch.float32, device=logits.device)

    grid = (seq_len, )

    BLOCK_SIZE_N = triton.next_power_of_2(vocab_size)
    if BLOCK_SIZE_N > 16384:
        BLOCK_SIZE_N = 16384
    
    _two_pass_kernel[grid](
        logits,
        output_indices,
        output_probs,
        seq_len,
        vocab_size,
        logits.stride(0),
        logits.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output_indices, output_probs