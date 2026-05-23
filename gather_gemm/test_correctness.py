#!/usr/bin/env python3
"""快速测试 gather_gemm 的正确性"""
import torch
from gather_gemm_kernel import gather_gemm

def test_gather_gemm():
    # 小规模测试
    Total_Len = 100
    Hidden = 4096
    Vocab = 1000
    M = 50
    
    # 创建测试数据
    x = torch.randn(Total_Len, Hidden, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(Vocab, Hidden, device='cuda', dtype=torch.bfloat16)
    indices = torch.randint(0, Total_Len, (M,), device='cuda', dtype=torch.int32)
    
    # 方法1: PyTorch gather + matmul (ground truth)
    x_gathered = x[indices.long()]
    out_torch = torch.matmul(x_gathered, weight.t())
    
    # 方法2: gather_gemm（注意：kernel 需要 weight.T）
    out_triton = torch.empty(M, Vocab, device='cuda', dtype=torch.bfloat16)
    gather_gemm(x, indices, weight.t(), out_triton)
    
    # 比较结果
    diff = (out_torch - out_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"out_torch sample: {out_torch[0, :10]}")
    print(f"out_triton sample: {out_triton[0, :10]}")
    
    # bfloat16 精度容忍度：max_diff < 1.0 且 mean_diff < 0.01
    if max_diff < 1.0 and mean_diff < 0.01:
        print("✅ PASS: gather_gemm 结果正确（在 bfloat16 精度范围内）")
        return True
    else:
        print("❌ FAIL: gather_gemm 结果不正确")
        return False

if __name__ == "__main__":
    test_gather_gemm()

