# file: benchmark.py
"""
基准测试脚本。
用于对比 Two-Pass Triton Kernel, Online Triton Kernel, 以及原生 PyTorch 实现
在计算 argmax 和对应 softmax 概率任务上的性能。
"""
import sys
sys.path.append('vllm_add_llada')
import torch
# 从另外两个文件中导入我们编写的函数
from vllm_add_llada.triton_kernels.two_pass_flash_sample import fused_argmax_softmax_twopass
from vllm_add_llada.triton_kernels.online_flash_sample import fused_argmax_softmax_online

def benchmark_pytorch(logits: torch.Tensor):
    """标准的 PyTorch 实现方式，用于验证和作为基线。"""
    # 注意：这会实例化一个完整的 softmax 概率矩阵，消耗大量显存
    probs = torch.softmax(logits.to(torch.float32), dim=1)
    indices = torch.argmax(logits, dim=1)
    # 使用 gather 提取在 argmax 索引处的概率值
    probs_at_indices = probs.gather(1, indices.unsqueeze(1)).squeeze(1)
    return indices, probs_at_indices

def run_benchmark():
    # --- 配置参数 ---
    SEQ_LEN = 80000
    VOCAB_SIZE = 126464
    DTYPE = torch.float16
    DEVICE = "cuda"

    if not torch.cuda.is_available():
        print("此脚本需要 CUDA GPU 才能运行。")
        return

    print("="*50)
    print(" fused argmax-softmax 性能基准测试")
    print("="*50)
    print(f"配置: seq_len={SEQ_LEN}, vocab_size={VOCAB_SIZE}, dtype={DTYPE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 创建测试数据 ---
    print("\n[1/4] 正在创建测试数据...")
    logits_tensor = torch.randn((SEQ_LEN, VOCAB_SIZE), device=DEVICE, dtype=DTYPE) * 5.0
    print("数据创建完毕。")

    # --- 运行并计时 ---
    implementations = {
        "PyTorch (原生)": benchmark_pytorch,
        "Triton (Two-Pass, 并行)": fused_argmax_softmax_twopass,
        "Triton (Online, 串行)": fused_argmax_softmax_online,
    }

    results = {}
    ground_truth = None

    for i, (name, func) in enumerate(implementations.items()):
        print(f"\n[{i+2}/4] 正在运行: {name}")
        
        # 预热
        for _ in range(10):
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

        if name == "PyTorch (原生)":
            ground_truth = (indices, probs)

    # --- 验证结果 ---
    print("\n[4/4] 正在验证结果...")
    is_v2_correct = (
        torch.allclose(results["Triton (Two-Pass, 并行)"]["indices"], ground_truth[0]) and
        torch.allclose(results["Triton (Two-Pass, 并行)"]["probs"], ground_truth[1], atol=1e-4)
    )
    is_online_correct = (
        torch.allclose(results["Triton (Online, 串行)"]["indices"], ground_truth[0]) and
        torch.allclose(results["Triton (Online, 串行)"]["probs"], ground_truth[1], atol=1e-4)
    )
    print(f"  - Two-Pass Triton 结果是否正确: {'是' if is_v2_correct else '否'}")
    print(f"  - Online Triton 结果是否正确:   {'是' if is_online_correct else '否'}")

    # --- 性能总结 ---
    print("\n" + "="*50)
    print(" 性能总结")
    print("="*50)

    pytorch_time = results["PyTorch (原生)"]["latency_ms"]
    twopass_time = results["Triton (Two-Pass, 并行)"]["latency_ms"]
    online_time = results["Triton (Online, 串行)"]["latency_ms"]

    print(f"PyTorch (原生)           : {pytorch_time:>8.4f} 毫秒")
    print(f"Triton (Two-Pass, 并行)  : {twopass_time:>8.4f} 毫秒")
    print(f"Triton (Online, 串行)    : {online_time:>8.4f} 毫秒")
    print("-"*50)
    print(f"性能提升 (Two-Pass vs PyTorch) : {pytorch_time / twopass_time:.2f}x")
    print(f"性能提升 (Two-Pass vs Online)  : {online_time / twopass_time:.2f}x")
    print("="*50)


if __name__ == "__main__":
    run_benchmark()