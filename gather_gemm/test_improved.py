import torch
import triton
import triton.language as tl
import numpy as np
import pandas as pd
from datetime import datetime

# ==========================================
# 1. Triton Kernel (Fused Gather-GEMM)
# ==========================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gather_matmul_kernel(
    x_ptr, w_ptr, indices_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid // group_size_m) % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_indices = tl.load(indices_ptr + offs_m, mask=offs_m < M, other=0)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 强制转为 int64 以防止溢出
    x_ptrs = x_ptr + (row_indices[:, None].to(tl.int64) * stride_xm) + (offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(x, w)
        
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # 强制转为 int64，这里是 100k 报错的核心原因
    out_ptrs = out_ptr + (stride_om * offs_m[:, None].to(tl.int64)) + (stride_on * offs_n[None, :].to(tl.int64))
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator.to(tl.float16), mask=c_mask)

def triton_mask_logits(hidden_states, indices, weight, M):
    _, Hidden = hidden_states.shape
    _, Vocab = weight.shape
    output = torch.empty((M, Vocab), device=hidden_states.device, dtype=hidden_states.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(Vocab, META['BLOCK_SIZE_N']), )
    gather_matmul_kernel[grid](
        hidden_states, weight, indices, output,
        M, Vocab, Hidden,
        hidden_states.stride(0), hidden_states.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    return output


# ==========================================
# 2. 改进的 Benchmark 函数
# ==========================================
def benchmark_method(func, warmup_iters=20, test_iters=100, **kwargs):
    """
    更严谨的 benchmark 函数
    
    Args:
        func: 要测试的函数
        warmup_iters: 预热次数
        test_iters: 测试次数
        **kwargs: 传递给 func 的参数
    """
    # 充分预热
    for _ in range(warmup_iters):
        func(**kwargs)
    torch.cuda.synchronize()
    
    # 正式测试
    times = []
    for _ in range(test_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = func(**kwargs)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # 去掉最大和最小的 10% 异常值
    times_sorted = sorted(times)
    trim_count = int(len(times) * 0.1)
    times_trimmed = times_sorted[trim_count:-trim_count] if trim_count > 0 else times_sorted
    
    return {
        'mean': np.mean(times_trimmed),
        'std': np.std(times_trimmed),
        'min': np.min(times_trimmed),
        'max': np.max(times_trimmed),
        'median': np.median(times_trimmed),
        'result': result
    }


# ==========================================
# 3. 测试函数定义
# ==========================================
def baseline_full(x, w, **kwargs):
    """全量计算"""
    return torch.matmul(x, w)

def pytorch_gather_gemm(x, w, indices, **kwargs):
    """PyTorch: Gather + GEMM"""
    x_gathered = x[indices.long()]
    return torch.matmul(x_gathered, w)

def triton_fused(x, w, indices, M, **kwargs):
    """Triton: Fused Kernel"""
    return triton_mask_logits(x, indices, w, M)


# ==========================================
# 4. 主测试程序
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU 才能运行")

    # 获取 GPU 信息
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print("=" * 70)

    # --- 配置参数 ---
    Hidden = 4096         # Hidden Size
    Vocab = 20000         # Vocab Size

    # 测试不同的序列长度
    seq_lengths = [4096, 10000, 50000, 100000]
    
    # 根据序列长度自适应调整迭代次数
    def get_iters_for_seq_len(seq_len):
        if seq_len <= 10000:
            return 20, 100  # warmup, test
        elif seq_len <= 50000:
            return 15, 50
        elif seq_len <= 100000:
            return 10, 30
        else:
            return 5, 20
    
    # 测试不同的 Mask Ratio
    mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    
    # 存储所有结果用于最后的总结
    all_results = []
    
    for Total_Len in seq_lengths:
        # 根据序列长度自适应调整迭代次数
        Warmup_Iters, Test_Iters = get_iters_for_seq_len(Total_Len)
        
        print(f"\n{'#'*70}")
        print(f"# TESTING SEQUENCE LENGTH: {Total_Len}")
        print(f"{'#'*70}")
        
        for Mask_Ratio in mask_ratios:
            print(f"\n{'='*70}")
            print(f"Config: L={Total_Len}, H={Hidden}, V={Vocab}, MaskRatio={Mask_Ratio}")
            print(f"Warmup: {Warmup_Iters} iters, Test: {Test_Iters} iters")
            print("-" * 70)
            
            # --- 数据构造 ---
            x = torch.randn(Total_Len, Hidden, device='cuda', dtype=torch.float16)
            w = torch.randn(Hidden, Vocab, device='cuda', dtype=torch.float16)
            
            # 构造 Mask 和 Indices
            mask_bool = torch.bernoulli(torch.full((Total_Len,), Mask_Ratio, device='cuda')).to(torch.bool)
            indices = torch.nonzero(mask_bool.view(-1), as_tuple=False).squeeze(1).to(torch.int32)
            M = indices.shape[0]
            
            print(f"Actual M (selected rows): {M} / {Total_Len} = {M/Total_Len:.1%}")
            
            # 准备参数
            common_kwargs = {'x': x, 'w': w, 'indices': indices, 'M': M}
            
            # ========================================================
            # 预先触发 Triton JIT 编译（不计入任何测试时间）
            # ========================================================
            print("\nTriggering Triton JIT compilation...")
            triton_mask_logits(x, indices, w, M)
            torch.cuda.synchronize()
            print("JIT compilation complete.")
            
            # ========================================================
            # 1. Baseline: Full Compute
            # ========================================================
            print("\n[1/3] Benchmarking Baseline (Full Compute)...")
            stats_baseline = benchmark_method(
                baseline_full, 
                warmup_iters=Warmup_Iters, 
                test_iters=Test_Iters,
                **common_kwargs
            )
            
            # ========================================================
            # 2. PyTorch: Gather + GEMM
            # ========================================================
            print("[2/3] Benchmarking PyTorch (Gather + GEMM)...")
            stats_pytorch = benchmark_method(
                pytorch_gather_gemm,
                warmup_iters=Warmup_Iters,
                test_iters=Test_Iters,
                **common_kwargs
            )
            
            # ========================================================
            # 3. Triton: Fused Kernel
            # ========================================================
            print("[3/3] Benchmarking Triton (Fused Kernel)...")
            stats_triton = benchmark_method(
                triton_fused,
                warmup_iters=Warmup_Iters,
                test_iters=Test_Iters,
                **common_kwargs
            )
            
            # ========================================================
            # 结果验证与总结
            # ========================================================
            y_torch = stats_pytorch['result']
            y_triton = stats_triton['result']
            diff = (y_torch - y_triton).abs().max().item()
            
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            
            print(f"\n{'Method':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
            print("-" * 70)
            print(f"{'Baseline (Full)':<20} {stats_baseline['mean']:>10.3f}   {stats_baseline['std']:>10.3f}   {stats_baseline['min']:>10.3f}   {stats_baseline['max']:>10.3f}")
            print(f"{'PyTorch (G+GEMM)':<20} {stats_pytorch['mean']:>10.3f}   {stats_pytorch['std']:>10.3f}   {stats_pytorch['min']:>10.3f}   {stats_pytorch['max']:>10.3f}")
            print(f"{'Triton (Fused)':<20} {stats_triton['mean']:>10.3f}   {stats_triton['std']:>10.3f}   {stats_triton['min']:>10.3f}   {stats_triton['max']:>10.3f}")
            
            print("\n" + "-" * 70)
            print(f"Correctness Check (Torch vs Triton):")
            print(f"  Max Diff: {diff:.4f} {'✅ PASS' if diff < 1e-1 else '❌ FAIL'}")
            
            print("\n" + "-" * 70)
            print(f"Speedup Analysis (relative to processing {M} rows):")
            
            # 理论计算量比例
            theoretical_ratio = M / Total_Len
            
            # PyTorch 相对于 Baseline
            pytorch_vs_baseline = stats_pytorch['mean'] / stats_baseline['mean']
            print(f"  PyTorch vs Baseline:    {pytorch_vs_baseline:.3f}x (理论: {theoretical_ratio:.3f}x)")
            
            # Triton 相对于 Baseline  
            triton_vs_baseline = stats_triton['mean'] / stats_baseline['mean']
            print(f"  Triton vs Baseline:     {triton_vs_baseline:.3f}x (理论: {theoretical_ratio:.3f}x)")
            
            # Triton 相对于 PyTorch
            speedup_triton_vs_pytorch = stats_pytorch['mean'] / stats_triton['mean']
            print(f"  Triton vs PyTorch:      {speedup_triton_vs_pytorch:.3f}x faster")
            
            # 计算效率（相对于理论最优）
            pytorch_efficiency = theoretical_ratio / pytorch_vs_baseline * 100
            triton_efficiency = theoretical_ratio / triton_vs_baseline * 100
            print(f"\n  Efficiency (vs theoretical best):")
            print(f"    PyTorch: {pytorch_efficiency:.1f}%")
            print(f"    Triton:  {triton_efficiency:.1f}%")
            
            # 存储结果
            all_results.append({
                'seq_len': Total_Len,
                'mask_ratio': Mask_Ratio,
                'actual_M': M,
                'baseline_time': stats_baseline['mean'],
                'pytorch_time': stats_pytorch['mean'],
                'triton_time': stats_triton['mean'],
                'speedup_triton_vs_pytorch': speedup_triton_vs_pytorch,
                'triton_efficiency': triton_efficiency
            })
            
            # 清理GPU缓存，避免OOM
            del x, w, indices, stats_baseline, stats_pytorch, stats_triton, y_torch, y_triton
            torch.cuda.empty_cache()
            print("✅ GPU cache cleared.")

    # ========================================================
    # 最终总结表格
    # ========================================================
    print("\n" + "#" * 70)
    print("# SUMMARY: All Results")
    print("#" * 70)
    print(f"\n{'SeqLen':<10} {'MaskR':<8} {'ActualM':<10} {'Baseline':<12} {'PyTorch':<12} {'Triton':<12} {'T/P':<8} {'Eff%':<8}")
    print("-" * 90)
    
    for result in all_results:
        print(f"{result['seq_len']:<10} "
              f"{result['mask_ratio']:<8.2f} "
              f"{result['actual_M']:<10} "
              f"{result['baseline_time']:<12.3f} "
              f"{result['pytorch_time']:<12.3f} "
              f"{result['triton_time']:<12.3f} "
              f"{result['speedup_triton_vs_pytorch']:<8.3f} "
              f"{result['triton_efficiency']:<8.1f}")
    
    # ========================================================
    # 导出结果到 XLSX
    # ========================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_filename = f"benchmark_results_{timestamp}.xlsx"
    
    # 构建DataFrame
    df = pd.DataFrame([{
        'SeqLen': result['seq_len'],
        'MaskRatio': result['mask_ratio'],
        'ActualM': result['actual_M'],
        'Baseline_ms': round(result['baseline_time'], 3),
        'PyTorch_ms': round(result['pytorch_time'], 3),
        'Triton_ms': round(result['triton_time'], 3),
        'Speedup_T/P': round(result['speedup_triton_vs_pytorch'], 3),
        'Efficiency_%': round(result['triton_efficiency'], 1)
    } for result in all_results])
    
    # 导出到Excel
    df.to_excel(xlsx_filename, index=False, engine='openpyxl')
    
    print(f"\n✅ Results exported to: {xlsx_filename}")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print("\nLegend:")
    print("  SeqLen   - Sequence Length")
    print("  MaskR    - Mask Ratio")
    print("  ActualM  - Actual number of masked tokens")
    print("  Baseline - Full compute time (ms)")
    print("  PyTorch  - Gather + GEMM time (ms)")
    print("  Triton   - Fused kernel time (ms)")
    print("  T/P      - Triton speedup vs PyTorch")
    print("  Eff%     - Triton efficiency vs theoretical best")

