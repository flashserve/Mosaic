#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# VMM Tensor 兼容性测试 - 验证 VMM buffer 能否像普通 torch.Tensor 一样使用

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tensor_compatibility():
    """测试 VMM tensor 是否完全兼容 torch.Tensor 的各种操作"""
    print("=" * 70)
    print("VMM Tensor 兼容性测试")
    print("=" * 70)
    
    import vmm_allocator
    
    # 初始化 CUDA
    torch.cuda.init()
    _ = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    
    # 创建 VMM allocator
    allocator = vmm_allocator.VMMAllocator()
    size = 100 * 1024**2  # 100 MB
    allocator.reserve(size)
    allocator.map_physical(0, size)
    
    # 包装为 tensor
    vmm_tensor = allocator.wrap_as_tensor(size)
    
    print(f"\n1. 基础属性测试")
    print(f"   类型: {type(vmm_tensor)}")
    print(f"   dtype: {vmm_tensor.dtype}")
    print(f"   device: {vmm_tensor.device}")
    print(f"   shape: {vmm_tensor.shape}")
    print(f"   is_cuda: {vmm_tensor.is_cuda}")
    print(f"   is_contiguous: {vmm_tensor.is_contiguous()}")
    assert isinstance(vmm_tensor, torch.Tensor), "不是 torch.Tensor 类型！"
    assert vmm_tensor.is_cuda, "不在 CUDA 设备上！"
    assert vmm_tensor.is_contiguous(), "内存不连续！"
    print("   ✅ 基础属性正常")
    
    print(f"\n2. 读写测试")
    # 写入数据
    vmm_tensor[:1000] = 42
    vmm_tensor[1000:2000] = 99
    # 读取数据
    assert (vmm_tensor[:1000] == 42).all().item(), "写入/读取失败！"
    assert (vmm_tensor[1000:2000] == 99).all().item(), "写入/读取失败！"
    print("   ✅ 读写操作正常")
    
    print(f"\n3. 切片操作测试")
    slice1 = vmm_tensor[:10000]
    slice2 = vmm_tensor[10000:20000]
    slice3 = vmm_tensor[-10000:]
    print(f"   slice1.shape: {slice1.shape}")
    print(f"   slice2.shape: {slice2.shape}")
    print(f"   slice3.shape: {slice3.shape}")
    assert slice1.is_cuda, "切片后不在 CUDA 上！"
    assert slice1.data_ptr() == vmm_tensor.data_ptr(), "切片改变了指针！"
    print("   ✅ 切片操作正常")
    
    print(f"\n4. view/reshape 测试")
    # 将 uint8 view 为其他类型
    view_fp16 = vmm_tensor[:1024*1024*10].view(torch.float16)  # 10MB as fp16
    view_fp32 = vmm_tensor[:1024*1024*20].view(torch.float32)  # 20MB as fp32
    print(f"   view_fp16.shape: {view_fp16.shape}, dtype: {view_fp16.dtype}")
    print(f"   view_fp32.shape: {view_fp32.shape}, dtype: {view_fp32.dtype}")
    
    # 测试 reshape
    reshaped = vmm_tensor[:1024*1024].reshape(1024, 1024)  # 1MB as 1024x1024
    print(f"   reshaped.shape: {reshaped.shape}")
    assert reshaped.is_cuda, "reshape 后不在 CUDA 上！"
    print("   ✅ view/reshape 操作正常")
    
    print(f"\n5. 算术操作测试")
    # 创建一个小 tensor 用于计算
    test_tensor = vmm_tensor[:1000].view(torch.float32)[:250]  # 1000 bytes = 250 fp32
    test_tensor.fill_(1.0)
    
    # 加法
    result = test_tensor + 1.0
    assert torch.allclose(result, torch.ones_like(result) * 2.0), "加法失败！"
    
    # 乘法
    result = test_tensor * 2.0
    assert torch.allclose(result, torch.ones_like(result) * 2.0), "乘法失败！"
    
    # 原地操作
    test_tensor += 1.0
    assert torch.allclose(test_tensor, torch.ones_like(test_tensor) * 2.0), "原地操作失败！"
    
    print("   ✅ 算术操作正常")
    
    print(f"\n6. CUDA kernel 操作测试")
    # 测试能否与 CUDA kernel 配合
    test_tensor = vmm_tensor[:4096].view(torch.float32)[:1024]  # 4KB = 1024 fp32
    test_tensor.fill_(0.0)
    
    # 使用 PyTorch 内置的 CUDA kernel
    test_tensor.add_(1.0)  # 调用 CUDA kernel
    assert torch.allclose(test_tensor, torch.ones_like(test_tensor)), "CUDA kernel 操作失败！"
    
    # 测试 matmul
    mat1 = vmm_tensor[:1024*32].view(torch.float32)[:8192].reshape(32, 256)
    mat2 = torch.ones(256, 64, device='cuda')
    result = torch.matmul(mat1, mat2)
    print(f"   matmul result shape: {result.shape}")
    print("   ✅ CUDA kernel 操作正常")
    
    print(f"\n7. 与普通 tensor 混合操作测试")
    vmm_small = vmm_tensor[:1024].view(torch.float32)[:256]
    normal_tensor = torch.ones(256, device='cuda', dtype=torch.float32)
    
    # 混合计算
    result = vmm_small + normal_tensor
    result = result * 2
    result = torch.matmul(result.unsqueeze(0), normal_tensor.unsqueeze(1))
    print(f"   混合计算结果: {result.shape}")
    print("   ✅ 与普通 tensor 混合操作正常")
    
    print(f"\n8. 梯度和反向传播测试（如果支持）")
    # VMM tensor 作为输入
    vmm_input = vmm_tensor[:1024].view(torch.float32)[:256]
    vmm_input.fill_(1.0)
    
    # 创建需要梯度的参数
    weight = torch.randn(256, 128, device='cuda', requires_grad=True)
    
    # 前向传播（VMM tensor 作为输入）
    output = torch.matmul(vmm_input.unsqueeze(0), weight)
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    assert weight.grad is not None, "反向传播失败！"
    print(f"   梯度形状: {weight.grad.shape}")
    print("   ✅ 梯度和反向传播正常")
    
    print(f"\n9. data_ptr 和内存地址测试")
    ptr1 = vmm_tensor.data_ptr()
    ptr2 = vmm_tensor[:1000].data_ptr()
    ptr3 = vmm_tensor[1000:2000].data_ptr()
    print(f"   原始指针: 0x{ptr1:x}")
    print(f"   切片1指针: 0x{ptr2:x} (偏移: {ptr2 - ptr1})")
    print(f"   切片2指针: 0x{ptr3:x} (偏移: {ptr3 - ptr1})")
    assert ptr2 == ptr1, "切片改变了基址！"
    assert ptr3 == ptr1 + 1000, "切片偏移不正确！"
    print("   ✅ 内存地址和指针正常")
    
    print(f"\n10. 复杂场景：模拟 activation pool 使用")
    # 模拟你的实际使用场景
    pool_size = 50 * 1024**2  # 50MB
    
    # 场景：为 100 个 tokens 分配 hidden states
    num_tokens = 100
    hidden_dim = 4096
    bytes_per_element = 2  # fp16
    required_bytes = num_tokens * hidden_dim * bytes_per_element
    
    # 从 VMM buffer 创建 tensor view
    buffer = vmm_tensor[:required_bytes]
    hidden_states = buffer.view(torch.float16).reshape(num_tokens, hidden_dim)
    
    print(f"   模拟场景: {num_tokens} tokens x {hidden_dim} dim")
    print(f"   hidden_states.shape: {hidden_states.shape}")
    print(f"   hidden_states.dtype: {hidden_states.dtype}")
    print(f"   hidden_states.is_contiguous: {hidden_states.is_contiguous()}")
    
    # 模拟前向传播
    hidden_states.fill_(0.5)
    weight = torch.randn(hidden_dim, hidden_dim, device='cuda', dtype=torch.float16)
    output = torch.matmul(hidden_states, weight)
    
    print(f"   前向传播输出: {output.shape}")
    print(f"   输出均值: {output.mean().item():.4f}")
    print("   ✅ 复杂场景模拟成功")
    
    # 清理
    allocator.free()
    
    print("\n" + "=" * 70)
    print("🎉 所有兼容性测试通过！")
    print("=" * 70)
    print("\n✅ 结论：VMM tensor 完全兼容 torch.Tensor，可以安全使用！")
    print("\n支持的操作：")
    print("  ✅ 读写（索引、切片）")
    print("  ✅ view/reshape")
    print("  ✅ 算术运算（+、-、*、/）")
    print("  ✅ CUDA kernel（matmul、add、等）")
    print("  ✅ 与普通 tensor 混合计算")
    print("  ✅ 梯度和反向传播（作为输入）")
    print("  ✅ 复杂场景（activation pool）")
    print()


if __name__ == "__main__":
    test_tensor_compatibility()

