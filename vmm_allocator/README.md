# CUDA VMM (Virtual Memory Management) Allocator

基于 CUDA Driver API 的虚拟内存管理分配器，用于实现零碎片、零浪费的 GPU 显存分配。

## 功能特性

- ✅ **零外部碎片**：虚拟地址空间连续
- ✅ **零内部浪费**：按需映射物理内存
- ✅ **动态映射**：根据实际需求逐步映射物理显存
- ✅ **PyTorch 集成**：零拷贝包装为 `torch.Tensor`

## 工作原理

```
传统 torch.empty(20GB):
┌────────────────────────────────────────────────┐
│  虚拟地址: 20GB (连续)                          │
│  物理显存: 20GB (立即分配)                      │
│  浪费: 如果只用 5GB，浪费 15GB ❌               │
└────────────────────────────────────────────────┘

VMM (本方案):
┌────────────────────────────────────────────────┐
│  虚拟地址: 20GB (连续) - reserve               │
│  物理显存: 0GB → 5GB (按需映射) - map          │
│  节省: 15GB 可供其他进程使用 ✅                 │
└────────────────────────────────────────────────┘
```

## 编译和安装

### 前置依赖

- CUDA >= 10.2 (支持 VMM API)
- PyTorch with CUDA
- GCC >= 7.0

### 编译

```bash
cd vmm_allocator
./build.sh
```

编译成功后会生成 `vmm_allocator.cpython-xxx.so` 文件。

### 测试

```bash
python test_vmm.py
```

## 使用示例

### 基础用法

```python
import vmm_allocator
import torch

# 创建 VMM 分配器
allocator = vmm_allocator.VMMAllocator()

# 1. 预留虚拟地址空间（不占用物理显存）
reserve_size = 20 * 1024**3  # 20 GB
allocator.reserve(reserve_size)

# 2. 按需映射物理内存
actual_size = 5 * 1024**3  # 实际只需要 5 GB
allocator.map_physical(0, actual_size)

# 3. 包装为 torch.Tensor（零拷贝）
tensor = allocator.wrap_as_tensor(actual_size)

# 4. 正常使用 tensor
tensor[:1000] = 42
print(tensor[:10])

# 5. 获取统计信息
stats = allocator.get_stats()
print(f"Reserved: {stats['reserved_size_mb']:.2f} MB")
print(f"Mapped: {stats['mapped_size_mb']:.2f} MB")
print(f"Utilization: {stats['utilization'] * 100:.1f}%")

# 6. 清理资源
allocator.free()
```

### 增量映射

```python
# 根据实际需求逐步增加物理内存映射
allocator.reserve(10 * 1024**3)  # 预留 10GB

# 第一次推理：需要 2GB
allocator.map_physical(0, 2 * 1024**3)

# 第二次推理：需要 5GB（自动增量映射）
allocator.map_physical(0, 5 * 1024**3)  # 只会映射新增的 3GB
```

## API 参考

### `VMMAllocator` 类

#### `reserve(size: int) -> int`
预留虚拟地址空间（不分配物理内存）。

- **参数**: `size` - 要预留的大小（字节）
- **返回**: 虚拟地址指针（整数）

#### `map_physical(offset: int, size: int) -> None`
映射物理内存到虚拟地址。

- **参数**:
  - `offset` - 虚拟地址偏移量
  - `size` - 要映射的大小（字节）

#### `wrap_as_tensor(size: int) -> torch.Tensor`
将 VMM 缓冲区包装为 PyTorch Tensor。

- **参数**: `size` - Tensor 大小（字节）
- **返回**: `torch.Tensor` (dtype=uint8, device=cuda)

#### `get_stats() -> dict`
获取内存统计信息。

- **返回**: 包含以下字段的字典：
  - `reserved_size_mb`: 已预留的虚拟地址空间（MB）
  - `mapped_size_mb`: 已映射的物理内存（MB）
  - `utilization`: 内存利用率（0-1）

#### `free() -> None`
释放所有 VMM 资源。

#### `get_granularity() -> int` (静态方法)
获取 VMM 页粒度（最小分配单位，通常为 2MB）。

## 性能考虑

### 优势

- **显存节省**: 预留大空间但只映射实际使用的部分
- **多租户**: 多个模型可以共享物理显存
- **灵活性**: 动态调整物理内存使用

### 权衡

- **映射延迟**: `cuMemMap` 调用有微秒到毫秒级开销
- **页对齐**: 实际分配会对齐到 2MB 边界（损耗 < 0.1%）

### 最佳实践

1. **分块映射**: 按 Chunk（如 1GB）为单位映射，而非按 Byte
2. **预测性映射**: 根据历史推理 token 数提前映射
3. **一次映射**: 避免频繁的增量映射

## 故障排查

### 编译错误

```
Error: CUDA Driver API not found
```

**解决**: 确保安装了 CUDA Toolkit 和 Driver，并设置 `CUDA_HOME` 环境变量。

### 运行时错误

```
CUDA_ERROR_NOT_SUPPORTED
```

**原因**: GPU 或 CUDA 版本不支持 VMM（需要 CUDA >= 10.2）

**解决**: 升级 CUDA 版本或使用传统的 `torch.empty`

## 许可证

Apache-2.0 License

