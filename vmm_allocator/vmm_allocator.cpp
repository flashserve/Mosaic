// SPDX-License-Identifier: Apache-2.0
// CUDA Virtual Memory Management (VMM) Allocator Implementation

#include "vmm_allocator.h"
#include <iostream>
#include <cstring>

namespace vmm {

VMMAllocator::VMMAllocator()
    : virtual_addr_(0),
      mem_handle_(0),
      reserved_size_(0),
      mapped_size_(0),
      granularity_(0),
      device_id_(0),
      initialized_(false) {
    
    // 初始化 CUDA Driver API
    CHECK_CUDA_DRIVER(cuInit(0));
    
    // 获取当前设备
    CHECK_CUDA_DRIVER(cuCtxGetDevice(&device_id_));
    
    // 获取页粒度
    granularity_ = get_granularity();
}

VMMAllocator::~VMMAllocator() {
    try {
        free();
    } catch (const std::exception& e) {
        std::cerr << "[VMMAllocator] Error in destructor: " << e.what() << std::endl;
    }
}

void* VMMAllocator::reserve(size_t reserve_size) {
    if (initialized_) {
        throw std::runtime_error("VMMAllocator already initialized");
    }

    // 对齐到页边界
    reserved_size_ = align_up(reserve_size, granularity_);
    
    std::cout << "[VMMAllocator] Reserving " << (reserved_size_ / (1024.0 * 1024.0 * 1024.0)) 
              << " GB of virtual address space (granularity: " << (granularity_ / (1024 * 1024)) 
              << " MB)" << std::endl;

    // 预留虚拟地址空间（不分配物理内存）
    CHECK_CUDA_DRIVER(cuMemAddressReserve(
        &virtual_addr_,
        reserved_size_,
        0,  // alignment (0 = use default)
        0,  // addr (0 = let CUDA choose)
        0   // flags
    ));

    initialized_ = true;
    mapped_size_ = 0;

    std::cout << "[VMMAllocator] Reserved virtual address: 0x" << std::hex << virtual_addr_ 
              << std::dec << std::endl;

    return get_ptr();
}

void VMMAllocator::map_physical(size_t offset, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("VMMAllocator not initialized");
    }

    // 对齐
    size_t aligned_offset = align_up(offset, granularity_);
    size_t aligned_size = align_up(size, granularity_);

    if (aligned_offset + aligned_size > reserved_size_) {
        throw std::runtime_error("Map size exceeds reserved size");
    }

    // 如果已经映射了这段内存，直接返回
    if (aligned_offset + aligned_size <= mapped_size_) {
        return;
    }

    // 计算需要新映射的大小
    size_t new_offset = mapped_size_;
    size_t new_size = aligned_offset + aligned_size - mapped_size_;

    std::cout << "[VMMAllocator] Mapping " << (new_size / (1024.0 * 1024.0)) 
              << " MB of physical memory at offset " << (new_offset / (1024.0 * 1024.0)) 
              << " MB" << std::endl;

    // 创建物理内存
    CUmemGenericAllocationHandle mem_handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id_;

    CHECK_CUDA_DRIVER(cuMemCreate(&mem_handle, new_size, &prop, 0));

    // 映射物理内存到虚拟地址
    CHECK_CUDA_DRIVER(cuMemMap(
        virtual_addr_ + new_offset,
        new_size,
        0,  // offset in physical memory
        mem_handle,
        0   // flags
    ));

    // 设置访问权限
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id_;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CHECK_CUDA_DRIVER(cuMemSetAccess(
        virtual_addr_ + new_offset,
        new_size,
        &access_desc,
        1
    ));

    // 更新已映射大小
    mapped_size_ = aligned_offset + aligned_size;

    std::cout << "[VMMAllocator] Total mapped: " << (mapped_size_ / (1024.0 * 1024.0 * 1024.0)) 
              << " GB / " << (reserved_size_ / (1024.0 * 1024.0 * 1024.0)) 
              << " GB (utilization: " << (100.0 * mapped_size_ / reserved_size_) << "%)" 
              << std::endl;

    // 注意：这里为了简化，我们不存储 mem_handle
    // 在生产环境中，应该存储所有 handle 以便后续释放
}

void VMMAllocator::unmap_physical(size_t offset, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("VMMAllocator not initialized");
    }

    // 对齐
    size_t aligned_offset = align_up(offset, granularity_);
    size_t aligned_size = align_up(size, granularity_);

    if (aligned_offset + aligned_size > mapped_size_) {
        return; // 没有映射，无需取消
    }

    // 取消映射
    CHECK_CUDA_DRIVER(cuMemUnmap(virtual_addr_ + aligned_offset, aligned_size));

    std::cout << "[VMMAllocator] Unmapped " << (aligned_size / (1024.0 * 1024.0)) 
              << " MB at offset " << (aligned_offset / (1024.0 * 1024.0)) << " MB" << std::endl;
}

void VMMAllocator::free() {
    if (!initialized_) {
        return;
    }

    std::cout << "[VMMAllocator] Freeing VMM resources" << std::endl;

    // 取消所有映射
    if (mapped_size_ > 0) {
        try {
            CHECK_CUDA_DRIVER(cuMemUnmap(virtual_addr_, mapped_size_));
        } catch (const std::exception& e) {
            std::cerr << "[VMMAllocator] Error unmapping: " << e.what() << std::endl;
        }
    }

    // 释放虚拟地址空间
    if (virtual_addr_ != 0) {
        try {
            CHECK_CUDA_DRIVER(cuMemAddressFree(virtual_addr_, reserved_size_));
        } catch (const std::exception& e) {
            std::cerr << "[VMMAllocator] Error freeing address: " << e.what() << std::endl;
        }
    }

    virtual_addr_ = 0;
    reserved_size_ = 0;
    mapped_size_ = 0;
    initialized_ = false;

    std::cout << "[VMMAllocator] Freed successfully" << std::endl;
}

size_t VMMAllocator::get_granularity() {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    
    int device_id;
    CHECK_CUDA_DRIVER(cuCtxGetDevice(&device_id));
    prop.location.id = device_id;

    size_t granularity;
    CHECK_CUDA_DRIVER(cuMemGetAllocationGranularity(
        &granularity,
        &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM
    ));

    return granularity;
}

size_t VMMAllocator::align_up(size_t size, size_t granularity) {
    return ((size + granularity - 1) / granularity) * granularity;
}

} // namespace vmm

