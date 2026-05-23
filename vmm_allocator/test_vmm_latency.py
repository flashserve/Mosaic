# test_vmm_latency.py
import torch
import time
import sys
sys.path.insert(0, 'vmm_allocator')

from vllm.v1.worker.activation_memory_pool import ActivationMemoryPool

def benchmark_mode(use_vmm, num_iterations=100):
    """对比两种模式的延迟"""
    
    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 创建 pool
    pool = ActivationMemoryPool(torch.device('cuda:0'), use_vmm=use_vmm)
    
    # 初始化
    t0 = time.perf_counter()
    class MockConfig:
        pass
    pool.pool_size = 10 * 1024**3  # 10GB
    pool._initialized = True
    if use_vmm:
        pool._allocate_vmm_pool()
    else:
        pool.pool_buffer = torch.empty(pool.pool_size, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    init_time = (time.perf_counter() - t0) * 1000
    
    # 推理延迟测试
    latencies = []
    sizes = [100 * 1024**2, 200 * 1024**2, 500 * 1024**2]  # 100MB, 200MB, 500MB
    
    for i in range(num_iterations):
        size = sizes[i % len(sizes)]
        
        t0 = time.perf_counter()
        buffer = pool.allocate_for_tokens_with_size(100, size)
        # 模拟使用
        buffer[:1000] = 42
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        
        latencies.append(latency)
    
    return init_time, latencies

# 测试
print("=" * 60)
print("VMM vs torch.empty 延迟对比")
print("=" * 60)

print("\n【torch.empty 模式】")
init_torch, latencies_torch = benchmark_mode(use_vmm=False, num_iterations=100)
print(f"初始化时间: {init_torch:.2f}ms")
print(f"首次推理: {latencies_torch[0]:.3f}ms")
print(f"平均延迟: {sum(latencies_torch[1:])/len(latencies_torch[1:]):.3f}ms")
print(f"P50 延迟: {sorted(latencies_torch)[50]:.3f}ms")
print(f"P99 延迟: {sorted(latencies_torch)[99]:.3f}ms")

print("\n【VMM 模式】")
init_vmm, latencies_vmm = benchmark_mode(use_vmm=True, num_iterations=100)
print(f"初始化时间: {init_vmm:.2f}ms ({'✅快' if init_vmm < init_torch else '❌慢'} {abs(init_torch/init_vmm):.1f}x)")
print(f"首次推理: {latencies_vmm[0]:.3f}ms (+{latencies_vmm[0]-latencies_torch[0]:.3f}ms)")
print(f"平均延迟: {sum(latencies_vmm[1:])/len(latencies_vmm[1:]):.3f}ms")
print(f"P50 延迟: {sorted(latencies_vmm)[50]:.3f}ms")
print(f"P99 延迟: {sorted(latencies_vmm)[99]:.3f}ms")

print("\n【结论】")
avg_diff = sum(latencies_vmm[10:]) / len(latencies_vmm[10:]) - sum(latencies_torch[10:]) / len(latencies_torch[10:])
print(f"稳定后延迟差异: {avg_diff:.3f}ms ({abs(avg_diff/sum(latencies_torch[10:])*len(latencies_torch[10:]))*100:.2f}%)")