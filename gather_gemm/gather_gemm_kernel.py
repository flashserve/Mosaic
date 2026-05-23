"""
Fused Gather-GEMM Triton Kernel
简洁接口：只需输入buffer、输出buffer、mask indices和权重
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['K'],  # 只基于 K（Hidden Size），所有 M/N 复用同一编译
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
    """
    Fused Gather-GEMM Kernel
    
    out[i] = x[indices[i]] @ w  for i in [0, M)
    
    Args:
        x_ptr: 输入矩阵 [Total_Len, K]
        w_ptr: 权重矩阵 [K, N]
        indices_ptr: 行索引 [M]
        out_ptr: 输出矩阵 [M, N]
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid // group_size_m) % num_pid_n

    # 加载行索引
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_indices = tl.load(indices_ptr + offs_m, mask=offs_m < M, other=0)
    
    # 确保 row_indices 在有效范围内（调试用，实际上应该由调用方保证）
    # row_indices = tl.where(offs_m < M, row_indices, 0)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算指针（使用 int64 避免大序列溢出）
    x_ptrs = x_ptr + (row_indices[:, None].to(tl.int64) * stride_xm) + (offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)

    # 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K 维度分块累加
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(x, w)
        
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # 存储结果（使用 int64 避免溢出）
    # Triton 会自动将 float32 accumulator 转换为输出 tensor 的 dtype (float16/bfloat16)
    out_ptrs = out_ptr + (stride_om * offs_m[:, None].to(tl.int64)) + (stride_on * offs_n[None, :].to(tl.int64))
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=c_mask)


def gather_gemm(x, indices, weight, out):
    """
    Fused Gather-GEMM 算子接口
    
    执行：out = x[indices] @ weight.T
    
    Args:
        x: 输入 tensor [Total_Len, Hidden], dtype=float16 或 bfloat16
        indices: 行索引 tensor [M], dtype=int32 或 int64
        weight: 权重 tensor [Vocab, Hidden], dtype=float16 或 bfloat16
        out: 输出 buffer [M, Vocab], dtype=float16 或 bfloat16（预分配，需与 x/weight 同 dtype）
    
    Returns:
        out: 填充后的输出 tensor [M, Vocab]
    
    Example:
        >>> # float16 示例
        >>> x = torch.randn(10000, 4096, device='cuda', dtype=torch.float16)
        >>> weight = torch.randn(32000, 4096, device='cuda', dtype=torch.float16)
        >>> indices = torch.randint(0, 10000, (500,), device='cuda', dtype=torch.int32)
        >>> out = torch.empty(500, 32000, device='cuda', dtype=torch.float16)
        >>> gather_gemm(x, indices, weight, out)
        
        >>> # bfloat16 示例
        >>> x_bf16 = torch.randn(10000, 4096, device='cuda', dtype=torch.bfloat16)
        >>> weight_bf16 = torch.randn(32000, 4096, device='cuda', dtype=torch.bfloat16)
        >>> out_bf16 = torch.empty(500, 32000, device='cuda', dtype=torch.bfloat16)
        >>> gather_gemm(x_bf16, indices, weight_bf16, out_bf16)
    """
    M = indices.shape[0]
    Hidden, Vocab = weight.shape  # weight 是 [Hidden, Vocab]（已转置）
    
    # 参数检查
    assert x.shape[1] == Hidden, f"x.shape[1]={x.shape[1]} != Hidden={Hidden}"
    assert out.shape == (M, Vocab), f"out.shape={out.shape} != ({M}, {Vocab})"
    assert x.dtype in (torch.float16, torch.bfloat16), f"x must be float16 or bfloat16, got {x.dtype}"
    assert weight.dtype in (torch.float16, torch.bfloat16), f"weight must be float16 or bfloat16, got {weight.dtype}"
    assert out.dtype in (torch.float16, torch.bfloat16), f"out must be float16 or bfloat16, got {out.dtype}"
    assert x.dtype == weight.dtype == out.dtype, f"All tensors must have same dtype, got x:{x.dtype}, weight:{weight.dtype}, out:{out.dtype}"
    assert x.is_cuda and weight.is_cuda and indices.is_cuda and out.is_cuda, "All tensors must be on CUDA"
    
    # 转换 indices 为 int32（如果需要）
    if indices.dtype == torch.int64:
        indices = indices.to(torch.int32)
    
    # 调用 kernel（使用 autotune，key=['K'] 避免每次 M 变化都重新编译）
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(Vocab, META['BLOCK_SIZE_N']), )
    gather_matmul_kernel[grid](
        x, weight, indices, out,
        M, Vocab, Hidden,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


def warmup_gather_gemm(hidden_size, vocab_size, device='cuda', warmup_sizes=[128, 512, 1024, 4096]):
    """
    预编译 gather_gemm kernel，避免首次运行时的 JIT 编译开销
    
    这个函数会触发 Triton autotune，编译并缓存所有配置。
    建议在模型初始化后、开始推理前调用一次。
    
    Args:
        hidden_size: 隐藏层大小（例如 4096）
        vocab_size: 词表大小（例如 32000）
        device: 设备（默认 'cuda'）
        warmup_sizes: 预热的 M 尺寸列表（默认覆盖常见大小）
    
    Example:
        >>> # 在模型加载后调用
        >>> warmup_gather_gemm(hidden_size=4096, vocab_size=32000)
        >>> print("Kernel precompiled and cached!")
    """
    print(f"[gather_gemm] 开始预编译 kernel (Hidden={hidden_size}, Vocab={vocab_size})...")
    
    # 创建虚拟数据
    total_len = max(warmup_sizes) * 2  # 确保足够大
    x_dummy = torch.randn(total_len, hidden_size, device=device, dtype=torch.float16)
    weight_dummy = torch.randn(vocab_size, hidden_size, device=device, dtype=torch.float16)
    
    # 对不同的 M 尺寸进行预热，触发 autotune
    for M in warmup_sizes:
        indices_dummy = torch.randint(0, total_len, (M,), device=device, dtype=torch.int32)
        out_dummy = torch.empty(M, vocab_size, device=device, dtype=torch.float16)
        
        # 触发编译
        gather_gemm(x_dummy, indices_dummy, weight_dummy, out_dummy)
        torch.cuda.synchronize()
        
        print(f"  ✓ Compiled for M={M}")
    
    print(f"[gather_gemm] 预编译完成！Kernel 已缓存，后续调用无编译开销。")
    
    # 清理显存
    del x_dummy, weight_dummy, indices_dummy, out_dummy
    torch.cuda.empty_cache()

