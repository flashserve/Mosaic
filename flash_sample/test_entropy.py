#!/usr/bin/env python3
"""
测试 flash_sample_entropy CUDA kernel
对比精度和性能与 PyTorch 原生实现
"""

import torch
import torch.nn.functional as F
import time
import flash_sample_entropy

def pytorch_entropy_baseline(logits):
    """
    PyTorch 原生实现（来自 dream_model.py）
    输入: logits [M, N] (bfloat16)
    输出: 
      - index [M]: argmax 索引
      - confidence [M]: 负熵
    """
    # Argmax
    _, x0_index = logits.max(dim=-1)  # [M]
    
    # Softmax
    p_2d = F.softmax(logits, dim=-1)  # [M, N]
    
    # Entropy: sum(p * log(p))
    epsilon = 1e-10
    log_probs = torch.log(p_2d + epsilon)  # [M, N]
    x0_confidence = torch.sum(p_2d * log_probs, dim=-1)  # [M] 负熵
    
    return x0_index, x0_confidence


def test_correctness(M, N, device='cuda'):
    """测试精度：对比 CUDA kernel 和 PyTorch 实现"""
    print(f"\n{'='*60}")
    print(f"精度测试: M={M}, N={N}")
    print(f"{'='*60}")
    
    # 生成随机 logits
    torch.manual_seed(42)
    logits = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    
    # PyTorch 实现
    logits_pt = logits.clone()
    idx_pt, conf_pt = pytorch_entropy_baseline(logits_pt)
    
    # CUDA kernel 实现
    logits_cuda = logits.clone().contiguous()
    idx_cuda, conf_cuda = flash_sample_entropy.entropy(logits_cuda)
    
    # 对比 index
    idx_match = torch.equal(idx_pt, idx_cuda)
    print(f"✓ Index 匹配: {idx_match}")
    if not idx_match:
        diff_mask = idx_pt != idx_cuda
        print(f"  不匹配数量: {diff_mask.sum().item()} / {M}")
        if diff_mask.sum() < 10:
            print(f"  不匹配位置: {torch.where(diff_mask)[0].tolist()}")
    
    # 对比 confidence (允许小误差)
    # 统一数据类型：转换为 float32
    conf_pt_f32 = conf_pt.float()
    conf_cuda_f32 = conf_cuda.float()
    
    conf_diff = torch.abs(conf_pt_f32 - conf_cuda_f32)
    max_diff = conf_diff.max().item()
    mean_diff = conf_diff.mean().item()
    
    print(f"✓ Confidence 差异:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    
    # 检查是否在合理误差范围内
    tolerance = 1e-2  # entropy 计算涉及更多浮点运算，容差稍大
    close_match = torch.allclose(conf_pt_f32, conf_cuda_f32, atol=tolerance, rtol=tolerance)
    print(f"✓ Confidence 接近 (tol={tolerance}): {close_match}")
    
    if not close_match:
        large_diff_mask = conf_diff > tolerance
        print(f"  大差异数量: {large_diff_mask.sum().item()} / {M}")
        if large_diff_mask.sum() < 5:
            indices = torch.where(large_diff_mask)[0][:5]
            for i in indices:
                print(f"    位置 {i}: PyTorch={conf_pt_f32[i]:.6f}, CUDA={conf_cuda_f32[i]:.6f}, diff={conf_diff[i]:.2e}")
    
    return idx_match and close_match


def benchmark_performance(M, N, device='cuda', warmup=10, repeat=100):
    """性能测试：对比运行时间"""
    print(f"\n{'='*60}")
    print(f"性能测试: M={M}, N={N}, warmup={warmup}, repeat={repeat}")
    print(f"{'='*60}")
    
    # 生成随机 logits
    torch.manual_seed(42)
    logits = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(warmup):
        _ = pytorch_entropy_baseline(logits.clone())
        _ = flash_sample_entropy.entropy(logits.clone().contiguous())
    
    torch.cuda.synchronize()
    
    # PyTorch 实现计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(repeat):
        _ = pytorch_entropy_baseline(logits.clone())
    end_event.record()
    torch.cuda.synchronize()
    
    pytorch_time = start_event.elapsed_time(end_event) / repeat  # ms
    
    # CUDA kernel 计时
    start_event.record()
    for _ in range(repeat):
        _ = flash_sample_entropy.entropy(logits.clone().contiguous())
    end_event.record()
    torch.cuda.synchronize()
    
    cuda_time = start_event.elapsed_time(end_event) / repeat  # ms
    
    # 结果
    speedup = pytorch_time / cuda_time
    print(f"PyTorch 实现: {pytorch_time:.3f} ms")
    print(f"CUDA Kernel:  {cuda_time:.3f} ms")
    print(f"加速比:       {speedup:.2f}x")
    
    return pytorch_time, cuda_time, speedup


def main():
    print("=" * 60)
    print("Flash Sample Entropy Kernel 测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 设备")
        return
    
    device = 'cuda'
    
    # 测试配置
    test_cases = [
        # (M, N)
        (128, 152064),    # 小 batch
        (512, 152064),    # 中 batch
        (1024, 152064),   # 大 batch (typical Dream vocab size)
        (2048, 152064),   # 超大 batch
    ]
    
    print(f"\nVocab Size: 152064 (Dream 模型)")
    print(f"数据类型: bfloat16")
    
    # 精度测试
    print("\n" + "=" * 60)
    print("第一部分: 精度验证")
    print("=" * 60)
    
    all_correct = True
    for M, N in test_cases[:2]:  # 只测试前两个，避免太慢
        correct = test_correctness(M, N, device)
        all_correct = all_correct and correct
    
    if all_correct:
        print(f"\n✅ 所有精度测试通过！")
    else:
        print(f"\n❌ 部分精度测试失败")
    
    # 性能测试
    print("\n" + "=" * 60)
    print("第二部分: 性能基准测试")
    print("=" * 60)
    
    results = []
    for M, N in test_cases:
        pt_time, cuda_time, speedup = benchmark_performance(M, N, device, warmup=10, repeat=100)
        results.append((M, pt_time, cuda_time, speedup))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'PyTorch (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for M, pt_time, cuda_time, speedup in results:
        print(f"{M:<12} {pt_time:<15.3f} {cuda_time:<15.3f} {speedup:<10.2f}x")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

