# file: benchmark.py
import torch
import math
import sys
import os

# =================================================================
# 【最终修复点】: 手动将包含 .so 文件的目录添加到 Python 的搜索路径中
# =================================================================
# 获取当前脚本文件(benchmark.py)所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将该路径插入到 Python 搜索路径的最前面
sys.path.insert(0, current_dir)
# =================================================================

# --- 步骤 1: 直接导入预编译好的模块 ---
# 因为我们已经把正确的路径告诉了 Python，所以这次一定能成功
try:
    import fused_ops
except ImportError as e:
    print("错误: 依然无法导入 'fused_ops' 模块。")
    print("请确保您已经执行了 'python setup.py build_ext --inplace' 并且在 'cuda_kernels' 目录下生成了 .so 文件。")
    print(f"原始报错信息: {e}")
    exit()

# --- 步骤 2: 定义 PyTorch 分块方案作为对比基线 ---
def pytorch_chunked_argmax_softmax(logits: torch.Tensor, chunk_size: int = 1024):
    """
    使用 PyTorch 原生操作，通过分块处理来高效、稳定地完成计算。
    这是我们用来验证正确性和对比性能的基准。
    """
    seq_len, _ = logits.shape
    output_indices = torch.empty(seq_len, dtype=torch.int64, device=logits.device)
    output_probs = torch.empty(seq_len, dtype=torch.float32, device=logits.device)
    num_chunks = math.ceil(seq_len / chunk_size)
    for i in range(num_chunks):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, seq_len)
        logits_chunk = logits[start_row:end_row, :]
        indices_chunk = torch.argmax(logits_chunk, dim=-1)
        probs_chunk = torch.softmax(logits_chunk.to(torch.float32), dim=-1)
        probs_at_indices_chunk = probs_chunk.gather(-1, indices_chunk.unsqueeze(-1)).squeeze(-1)
        output_indices[start_row:end_row] = indices_chunk
        output_probs[start_row:end_row] = probs_at_indices_chunk
    return output_indices, output_probs

# --- 步骤 3: 运行基准测试 ---
def run_benchmark():
    """主函数，执行所有测试、验证和性能总结。"""
    # --- 配置参数 ---
    SEQ_LEN = 60000 
    VOCAB_SIZE = 126464
    DTYPE = torch.float16
    DEVICE = "cuda"

    print("\n" + "="*50)
    print(" 高效 CUDA vs PyTorch 分块方案 性能基准测试")
    print("="*50)
    print(f"配置: seq_len={SEQ_LEN}, vocab_size={VOCAB_SIZE}, dtype={DTYPE}")
    
    print("\n[1/2] 正在创建测试数据...")
    logits_tensor = torch.randn((SEQ_LEN, VOCAB_SIZE), device=DEVICE, dtype=DTYPE)
    print("数据创建完毕。")

    implementations = {
        "PyTorch (分块)": pytorch_chunked_argmax_softmax,
        "自定义 CUDA (高效版)": lambda t: fused_ops.forward(t),
    }

    results = {}
    ground_truth = None

    for name, func in implementations.items():
        print(f"\n[Running] 正在运行: {name}")
        # 预热
        for _ in range(5):
            _ = func(logits_tensor)
        torch.cuda.synchronize()

        # 正式计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        indices, probs = func(logits_tensor)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        
        results[name] = {"latency_ms": latency_ms, "indices": indices, "probs": probs}
        print(f"  -> 完成，耗时: {latency_ms:.4f} 毫秒")

        if name == "PyTorch (分块)":
            ground_truth = (indices, probs)

    print("\n" + "="*50)
    print(" 结果验证与总结")
    print("="*50)

    is_cuda_correct = (
        torch.allclose(results["自定义 CUDA (高效版)"]["indices"], ground_truth[0]) and
        torch.allclose(results["自定义 CUDA (高效版)"]["probs"], ground_truth[1], atol=1e-3)
    )
    print(f"自定义 CUDA (高效版) 结果是否正确: {'是' if is_cuda_correct else '否'}")

    pytorch_time = results["PyTorch (分块)"]["latency_ms"]
    cuda_time = results["自定义 CUDA (高效版)"]["latency_ms"]
    print(f"\nPyTorch (分块)        : {pytorch_time:.4f} 毫秒")
    print(f"自定义 CUDA (高效版) : {cuda_time:.4f} 毫秒")
    print(f"性能提升 (CUDA vs PyTorch 分块): {pytorch_time / cuda_time:.2f}x")
    print("="*50)

if __name__ == '__main__':
    run_benchmark()