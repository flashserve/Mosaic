import torch
import triton
import triton.language as tl

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
    M, N, K, Total_Rows,
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
    # 关键：加载 Indices
    row_indices = tl.load(indices_ptr + offs_m, mask=offs_m < M, other=0)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 关键：间接寻址计算 x_ptrs
    x_ptrs = x_ptr + (row_indices[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load Indirect
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        # Load Direct
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(x, w)
        
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    out_ptrs = out_ptr + (stride_om * offs_m[:, None]) + (stride_on * offs_n[None, :])
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
# 2. 性能测试主程序
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU 才能运行")

    # --- 配置参数 (模拟 Llama-3-8B 规模) ---
    Total_Len = 4096      # 序列长度
    Hidden = 4096         # Hidden Size
    Vocab = 32000         # Vocab Size
    Mask_Ratio = 0.9      # Mask 比例
    Iter_Count = 100      # 循环次数，取平均值更准

    print(f"Config: L={Total_Len}, H={Hidden}, V={Vocab}, MaskRatio={Mask_Ratio}")
    print(f"Running {Iter_Count} iterations for timing...")
    print("-" * 60)
    
    # --- 数据构造 ---
    x = torch.randn(Total_Len, Hidden, device='cuda', dtype=torch.float16)
    w = torch.randn(Hidden, Vocab, device='cuda', dtype=torch.float16)
    
    # 构造 Mask 和 Indices
    mask_bool = torch.bernoulli(torch.full((Total_Len,), Mask_Ratio, device='cuda')).to(torch.bool)
    indices = torch.nonzero(mask_bool.view(-1), as_tuple=False).squeeze(1).to(torch.int32)
    M = indices.shape[0]

    # --- 预热 (Warmup) ---
    for _ in range(10):
        torch.matmul(x[:100], w)
    torch.cuda.synchronize()

    # ========================================================
    # 1. Baseline: Full Compute (算全量)
    # ========================================================
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(Iter_Count):
        # 纯算全量，没有任何 gather 开销
        y_full = torch.matmul(x, w)
    end.record()
    
    torch.cuda.synchronize()
    time_baseline = start.elapsed_time(end) / Iter_Count
    print(f"[Baseline] Full Compute:  {time_baseline:.3f} ms")

    # ========================================================
    # 2. PyTorch: Index Select + Matmul
    # ========================================================
    start.record()
    for _ in range(Iter_Count):
        # 包含 Gather 和 Compute 的总时间
        x_gathered = x[indices.long()]
        y_torch = torch.matmul(x_gathered, w)
    end.record()
    
    torch.cuda.synchronize()
    time_torch = start.elapsed_time(end) / Iter_Count
    print(f"[PyTorch ] Gather + GEMM: {time_torch:.3f} ms")

    # ========================================================
    # 3. Triton: Fused Indirect
    # ========================================================
    # 先跑一次触发 JIT 编译，不计入时间
    triton_mask_logits(x, indices, w, M)
    torch.cuda.synchronize()

    start.record()
    for _ in range(Iter_Count):
        y_triton = triton_mask_logits(x, indices, w, M)
    end.record()
    
    torch.cuda.synchronize()
    time_triton = start.elapsed_time(end) / Iter_Count
    print(f"[Triton  ] Fused Kernel:  {time_triton:.3f} ms")

    # ========================================================
    # 结果验证与总结
    # ========================================================
    # 简单验证一下 PyTorch 和 Triton 结果是否一致
    diff = (y_torch - y_triton).abs().max().item()
    
    print("-" * 60)
    print(f"Correctness Diff (Torch vs Triton): {diff:.4f} " + ("✅" if diff < 1e-1 else "❌"))
    print("-" * 60)
    print(f"Speedup vs Baseline:")
    print(f"PyTorch Gather: {time_baseline / time_torch:.2f}x")
    print(f"Triton Fused:   {time_baseline / time_triton:.2f}x")