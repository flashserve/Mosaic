// SPDX-License-Identifier: Apache-2.0
// CUDA Virtual Memory Management (VMM) Allocator
// 用于实现零碎片的显存分配

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace vmm {

// CUDA Driver API 错误检查宏
#define CHECK_CUDA_DRIVER(call)                                                \
    do {                                                                       \
        CUresult result = (call);                                              \
        if (result != CUDA_SUCCESS) {                                          \
            const char* err_name = nullptr;                                    \
            const char* err_string = nullptr;                                  \
            cuGetErrorName(result, &err_name);                                 \
            cuGetErrorString(result, &err_string);                             \
            throw std::runtime_error(std::string("CUDA Driver API error: ") +  \
                                   std::string(err_name ? err_name : "unknown") + \
                                   " - " +                                     \
                                   std::string(err_string ? err_string : "unknown")); \
        }                                                                      \
    } while (0)

class VMMAllocator {
public:
    VMMAllocator();
    ~VMMAllocator();

    // 预留虚拟地址空间（不分配物理内存）
    // reserve_size: 要预留的虚拟地址空间大小（字节）
    // 返回值: 虚拟地址指针
    void* reserve(size_t reserve_size);

    // 映射物理内存到虚拟地址空间
    // offset: 从虚拟地址起始的偏移量
    // size: 要映射的大小（字节）
    void map_physical(size_t offset, size_t size);

    // 取消映射物理内存
    // offset: 从虚拟地址起始的偏移量
    // size: 要取消映射的大小（字节）
    void unmap_physical(size_t offset, size_t size);

    // 释放所有资源
    void free();

    // 获取虚拟地址指针
    void* get_ptr() const { return reinterpret_cast<void*>(virtual_addr_); }

    // 获取已预留的大小
    size_t get_reserved_size() const { return reserved_size_; }

    // 获取已映射的大小
    size_t get_mapped_size() const { return mapped_size_; }

    // 获取页粒度（CUDA VMM 的最小分配单位）
    static size_t get_granularity();

    // 对齐到页边界
    static size_t align_up(size_t size, size_t granularity);

private:
    CUdeviceptr virtual_addr_;           // 虚拟地址
    CUmemGenericAllocationHandle mem_handle_; // 物理内存句柄
    size_t reserved_size_;               // 已预留的虚拟地址空间大小
    size_t mapped_size_;                 // 已映射的物理内存大小
    size_t granularity_;                 // 页粒度
    int device_id_;                      // CUDA 设备 ID
    bool initialized_;                   // 是否已初始化
};

} // namespace vmm

