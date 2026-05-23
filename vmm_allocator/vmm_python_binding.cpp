// SPDX-License-Identifier: Apache-2.0
// Python bindings for CUDA VMM Allocator

#include <torch/extension.h>
#include "vmm_allocator.h"
#include <pybind11/pybind11.h>
#include <memory>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/StorageImpl.h>

namespace py = pybind11;

// Python 包装类
class VMMAllocatorPython {
public:
    VMMAllocatorPython() : allocator_(std::make_shared<vmm::VMMAllocator>()) {}

    // 预留虚拟地址空间
    uintptr_t reserve(size_t size) {
        void* ptr = allocator_->reserve(size);
        return reinterpret_cast<uintptr_t>(ptr);
    }

    // 映射物理内存
    void map_physical(size_t offset, size_t size) {
        allocator_->map_physical(offset, size);
    }

    // 取消映射
    void unmap_physical(size_t offset, size_t size) {
        allocator_->unmap_physical(offset, size);
    }

    // 释放资源
    void free() {
        allocator_->free();
    }

    // 包装成 torch.Tensor
    torch::Tensor wrap_as_tensor(size_t size) {
        CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(allocator_->get_ptr());
        if (device_ptr == 0) {
            throw std::runtime_error("VMMAllocator not initialized");
        }

        // 使用底层 C10 API 创建 DataPtr 来包装 CUDA 设备指针
        auto data_ptr = c10::InefficientStdFunctionContext::makeDataPtr(
            reinterpret_cast<void*>(device_ptr),
            [](void*) {
                // 空 deleter - 我们手动管理 VMM 内存
            },
            c10::Device(c10::DeviceType::CUDA, 0)
        );

        // 创建 Storage
        auto storage = c10::Storage(
            c10::Storage::use_byte_size_t(),
            size,
            std::move(data_ptr),
            /*allocator=*/nullptr,
            /*resizable=*/false
        );

        // 创建 Tensor
        auto options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA, 0);

        auto tensor = torch::empty({0}, options);
        tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(storage);
        
        // 明确使用 IntArrayRef 版本避免重载歧义
        std::vector<int64_t> sizes = {static_cast<int64_t>(size)};
        std::vector<int64_t> strides = {1};
        tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            c10::IntArrayRef(sizes),
            c10::IntArrayRef(strides)
        );
        
        return tensor;
    }

    // 获取状态信息
    py::dict get_stats() {
        py::dict stats;
        stats["reserved_size_mb"] = allocator_->get_reserved_size() / (1024.0 * 1024.0);
        stats["mapped_size_mb"] = allocator_->get_mapped_size() / (1024.0 * 1024.0);
        stats["utilization"] = allocator_->get_reserved_size() > 0 
            ? static_cast<double>(allocator_->get_mapped_size()) / allocator_->get_reserved_size()
            : 0.0;
        return stats;
    }

    static size_t get_granularity() {
        return vmm::VMMAllocator::get_granularity();
    }

private:
    std::shared_ptr<vmm::VMMAllocator> allocator_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Virtual Memory Management (VMM) Allocator for PyTorch";

    py::class_<VMMAllocatorPython>(m, "VMMAllocator")
        .def(py::init<>())
        .def("reserve", &VMMAllocatorPython::reserve,
             py::arg("size"),
             "Reserve virtual address space (does not allocate physical memory)")
        .def("map_physical", &VMMAllocatorPython::map_physical,
             py::arg("offset"), py::arg("size"),
             "Map physical memory to virtual address space")
        .def("unmap_physical", &VMMAllocatorPython::unmap_physical,
             py::arg("offset"), py::arg("size"),
             "Unmap physical memory from virtual address space")
        .def("free", &VMMAllocatorPython::free,
             "Free all VMM resources")
        .def("wrap_as_tensor", &VMMAllocatorPython::wrap_as_tensor,
             py::arg("size"),
             "Wrap VMM buffer as PyTorch tensor (zero-copy)")
        .def("get_stats", &VMMAllocatorPython::get_stats,
             "Get VMM statistics")
        .def_static("get_granularity", &VMMAllocatorPython::get_granularity,
                   "Get VMM page granularity (minimum allocation unit)");
}

