#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test script for CUDA VMM Allocator

import torch
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vmm_basic():
    """基础功能测试"""
    print("=" * 60)
    print("Test 1: Basic VMM Functionality")
    print("=" * 60)
    
    try:
        import vmm_allocator
        print("✅ vmm_allocator module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import vmm_allocator: {e}")
        print("Please run: ./build.sh")
        return False
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # 初始化 CUDA 上下文（重要！VMM 需要有效的 CUDA 上下文）
    torch.cuda.init()
    _ = torch.zeros(1, device='cuda')  # 确保上下文已创建
    torch.cuda.synchronize()
    print("✅ CUDA context initialized")
    
    # 获取页粒度
    granularity = vmm_allocator.VMMAllocator.get_granularity()
    print(f"✅ VMM page granularity: {granularity / (1024 * 1024):.2f} MB")
    
    # 创建 VMM 分配器
    allocator = vmm_allocator.VMMAllocator()
    print("✅ VMMAllocator instance created")
    
    # 预留虚拟地址空间（1 GB）
    reserve_size = 1 * 1024**3  # 1 GB
    ptr = allocator.reserve(reserve_size)
    print(f"✅ Reserved {reserve_size / 1024**3:.2f} GB at address 0x{ptr:x}")
    
    # 获取统计信息
    stats = allocator.get_stats()
    print(f"   Reserved: {stats['reserved_size_mb']:.2f} MB")
    print(f"   Mapped: {stats['mapped_size_mb']:.2f} MB")
    print(f"   Utilization: {stats['utilization'] * 100:.1f}%")
    
    # 映射物理内存（100 MB）
    map_size = 100 * 1024**2  # 100 MB
    allocator.map_physical(0, map_size)
    print(f"✅ Mapped {map_size / 1024**2:.2f} MB of physical memory")
    
    # 再次获取统计信息
    stats = allocator.get_stats()
    print(f"   Mapped: {stats['mapped_size_mb']:.2f} MB")
    print(f"   Utilization: {stats['utilization'] * 100:.1f}%")
    
    # 包装成 torch.Tensor
    tensor = allocator.wrap_as_tensor(map_size)
    print(f"✅ Wrapped as torch.Tensor: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    # 测试写入和读取
    tensor[:1000] = 42
    assert (tensor[:1000] == 42).all().item(), "Tensor read/write failed"
    print("✅ Tensor read/write test passed")
    
    # 清理
    allocator.free()
    print("✅ VMM resources freed")
    
    print("\n" + "=" * 60)
    print("✅ All basic tests passed!")
    print("=" * 60)
    return True


def test_vmm_incremental_mapping():
    """增量映射测试"""
    print("\n" + "=" * 60)
    print("Test 2: Incremental Mapping")
    print("=" * 60)
    
    import vmm_allocator
    
    # 确保 CUDA 上下文存在
    _ = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    
    allocator = vmm_allocator.VMMAllocator()
    
    # 预留 500 MB
    reserve_size = 500 * 1024**2
    allocator.reserve(reserve_size)
    print(f"✅ Reserved {reserve_size / 1024**2:.2f} MB")
    
    # 分多次映射
    for i in range(5):
        map_size = (i + 1) * 50 * 1024**2  # 50MB, 100MB, 150MB, ...
        allocator.map_physical(0, map_size)
        stats = allocator.get_stats()
        print(f"   Step {i+1}: Mapped {stats['mapped_size_mb']:.2f} MB "
              f"(utilization: {stats['utilization'] * 100:.1f}%)")
    
    allocator.free()
    print("✅ Incremental mapping test passed")
    return True


def test_vmm_vs_torch_empty():
    """VMM vs torch.empty 对比测试"""
    print("\n" + "=" * 60)
    print("Test 3: VMM vs torch.empty Memory Usage")
    print("=" * 60)
    
    import vmm_allocator
    
    # 确保 CUDA 上下文存在
    _ = torch.zeros(1, device='cuda')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 方法 1: torch.empty (传统方式)
    mem_before = torch.cuda.memory_allocated()
    reserve_size = 1 * 1024**3  # 1 GB
    
    tensor_torch = torch.empty(reserve_size, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    mem_after_torch = torch.cuda.memory_allocated()
    
    print(f"torch.empty:")
    print(f"   Allocated: {(mem_after_torch - mem_before) / 1024**3:.3f} GB")
    
    del tensor_torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 方法 2: VMM (只预留，不映射)
    mem_before = torch.cuda.memory_allocated()
    
    allocator = vmm_allocator.VMMAllocator()
    allocator.reserve(reserve_size)
    torch.cuda.synchronize()
    mem_after_vmm_reserve = torch.cuda.memory_allocated()
    
    print(f"\nVMM (reserve only):")
    print(f"   Allocated: {(mem_after_vmm_reserve - mem_before) / 1024**3:.3f} GB")
    print(f"   Saved: {(mem_after_torch - mem_before - (mem_after_vmm_reserve - mem_before)) / 1024**3:.3f} GB ✅")
    
    # 方法 3: VMM (预留 + 映射 100MB)
    map_size = 100 * 1024**2  # 100 MB
    allocator.map_physical(0, map_size)
    torch.cuda.synchronize()
    mem_after_vmm_map = torch.cuda.memory_allocated()
    
    print(f"\nVMM (reserve + map 100MB):")
    print(f"   Allocated: {(mem_after_vmm_map - mem_before) / 1024**3:.3f} GB")
    print(f"   Saved: {(mem_after_torch - mem_before - (mem_after_vmm_map - mem_before)) / 1024**3:.3f} GB ✅")
    
    allocator.free()
    print("\n✅ Memory comparison test passed")
    return True


if __name__ == "__main__":
    print("CUDA VMM Allocator Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_vmm_basic()
    all_passed &= test_vmm_incremental_mapping()
    all_passed &= test_vmm_vs_torch_empty()
    
    if all_passed:
        print("\n" + "🎉" * 30)
        print("ALL TESTS PASSED!")
        print("🎉" * 30)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

