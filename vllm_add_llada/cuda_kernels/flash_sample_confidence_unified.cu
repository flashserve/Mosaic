// file: flash_sample_confidence_unified.cu
#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>

// =================================================================
// 1. CUDA Kernel 部分
// =================================================================
// 设备端标量到 float 的转换重载（避免 C++17 if constexpr 依赖）
__device__ __forceinline__ float scalar_to_float(half v) {
    return __half2float(v);
}

__device__ __forceinline__ float scalar_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <int BLOCK_SIZE, typename ScalarT>
__global__ void fused_argmax_softmax_kernel_high_perf(
    const ScalarT* logits,
    long long* out_indices,
    float* out_probs,
    long long M,
    long long N
) {
    long long row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= M) {
        return;
    }

    const ScalarT* row_logits = logits + row_idx * N;
    
    // 步骤 1: 线程级归约
    float thread_max_val = -FLT_MAX;
    long long thread_argmax_idx = -1;

    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float val = scalar_to_float(row_logits[col_idx]);
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_argmax_idx = col_idx;
        }
    }

    // 步骤 2: 块内归约 (寻找全局最大值/索引)
    __shared__ float s_vals[BLOCK_SIZE];
    __shared__ long long s_idxs[BLOCK_SIZE];

    s_vals[tid] = thread_max_val;
    s_idxs[tid] = thread_argmax_idx;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            float other_val = s_vals[tid + offset];
            long long other_idx = s_idxs[tid + offset];

            // 【最终精度修复点】: 在这里处理最大值相等的情况
            if (other_val > s_vals[tid]) {
                // 如果另一个值更大，则更新值和索引
                s_vals[tid] = other_val;
                s_idxs[tid] = other_idx;
            } else if (other_val == s_vals[tid]) {
                // 如果值相等，我们选择索引更小的那个，以匹配 PyTorch 的行为
                s_idxs[tid] = min(s_idxs[tid], other_idx);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_vals[BLOCK_SIZE - 1] = s_vals[0];
        s_idxs[BLOCK_SIZE - 1] = s_idxs[0];
    }
    __syncthreads();
    
    float final_max_val = s_vals[BLOCK_SIZE - 1];
    long long final_argmax_idx = s_idxs[BLOCK_SIZE - 1];
    
    // 步骤 3: 计算 Softmax 分母
    float thread_denominator = 0.0f;
    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float val = scalar_to_float(row_logits[col_idx]);
        thread_denominator += expf(val - final_max_val);
    }
    
    s_vals[tid] = thread_denominator;
    __syncthreads();
    
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            s_vals[tid] += s_vals[tid + offset];
        }
        __syncthreads();
    }

    // 步骤 4: 线程 0 将最终结果写回全局显存
    if (tid == 0) {
        float final_denominator = s_vals[0];
        out_probs[row_idx] = 1.0f / final_denominator;
        out_indices[row_idx] = final_argmax_idx;
    }
}


// =================================================================
// 2. C++ 启动器部分
// =================================================================
template <typename ScalarT>
void launch_fused_kernel_typed(
    const ScalarT* logits,
    long long* out_indices,
    float* out_probs,
    long long M,
    long long N,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 1024;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    
    fused_argmax_softmax_kernel_high_perf<BLOCK_SIZE, ScalarT><<<grid, block, 0, stream>>>(
        logits,
        out_indices,
        out_probs,
        M,
        N
    );
}




// =================================================================
// 3. PyTorch 绑定部分
// =================================================================
std::vector<torch::Tensor> fused_argmax_softmax_cuda(
    torch::Tensor logits
) {
    TORCH_CHECK(logits.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(logits.dim() == 2, "输入张量必须是 2D");
    TORCH_CHECK(logits.is_contiguous(), "输入张量必须是连续的");
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat16 || logits.scalar_type() == torch::kBFloat16,
        "输入张量必须是 Float16 或 BFloat16 类型"
    );

    const long long M = logits.size(0);
    const long long N = logits.size(1);

    auto out_indices = torch::empty({M}, logits.options().dtype(torch::kInt64));
    auto out_probs = torch::empty({M}, logits.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (logits.scalar_type() == torch::kFloat16) {
        launch_fused_kernel_typed<half>(
            reinterpret_cast<const half*>(logits.data_ptr()),
            (long long*)out_indices.data_ptr(),
            (float*)out_probs.data_ptr(),
            M,
            N,
            stream
        );
    } else {
        launch_fused_kernel_typed<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            (long long*)out_indices.data_ptr(),
            (float*)out_probs.data_ptr(),
            M,
            N,
            stream
        );
    }
    
    return {out_indices, out_probs};
}







#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h> // 确保包含了 bfloat16 的头文件
#include <c10/cuda/CUDAStream.h> // for c10::cuda::getCurrentCUDAStream

// =================================================================
// SiLU Kernel (In-place, BF16, Vectorized) - 最终修正版
// =================================================================
template <int BLOCK_SIZE>
__global__ void silu_inplace_kernel_bf16(
    __nv_bfloat16* data,
    long long N
) {
    // 循环应该遍历向量的索引，而不是元素的索引。
    // 每个线程处理一个向量（4个bf16）。
    const int VEC_SIZE = 4;
    long long num_vectors = N / VEC_SIZE;

    // 线程ID和步长现在是基于向量来计算的
    long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_stride = gridDim.x * blockDim.x;

    for (long long vec_idx = thread_id; vec_idx < num_vectors; vec_idx += grid_stride) {
        // 从向量索引计算出在原始数据中的元素索引
        // 这个 data_idx 保证是 VEC_SIZE (4) 的倍数，从而保证了内存对齐
        long long data_idx = vec_idx * VEC_SIZE;

        // 使用正确的向量类型 uint2 一次性加载 4 个 bf16
        // 4个 bfloat16 = 8 字节 = 64 位 = uint2 的大小
        uint2 data_vec = *reinterpret_cast<uint2*>(&data[data_idx]);

        // 使用高效的 intrinsics 进行数据转换
        // 将 uint2 视为两个 bfloat162
        __nv_bfloat162* bf162_ptr = reinterpret_cast<__nv_bfloat162*>(&data_vec);
        // 一次性将 bfloat162 转换为 float2
        float2 f2_1 = __bfloat1622float2(bf162_ptr[0]);
        float2 f2_2 = __bfloat1622float2(bf162_ptr[1]);

        // 在 float32 精度下进行计算
        f2_1.x = f2_1.x / (1.0f + expf(-f2_1.x));
        f2_1.y = f2_1.y / (1.0f + expf(-f2_1.y));
        f2_2.x = f2_2.x / (1.0f + expf(-f2_2.x));
        f2_2.y = f2_2.y / (1.0f + expf(-f2_2.y));

        // 将 float2 转回 bfloat162
        bf162_ptr[0] = __float22bfloat162_rn(f2_1);
        bf162_ptr[1] = __float22bfloat162_rn(f2_2);

        // 一次性将修改后的 uint2 写回全局内存
        *reinterpret_cast<uint2*>(&data[data_idx]) = data_vec;
    }

    // 处理剩余元素的循环（当N不能被4整除时）
    long long aligned_size = num_vectors * VEC_SIZE;
    for (long long idx = aligned_size + thread_id; idx < N; idx += grid_stride) {
        float x = __bfloat162float(data[idx]);
        float y = x / (1.0f + expf(-x));
        data[idx] = __float2bfloat16(y);
    }
}


// SiLU kernel 启动器
void launch_silu_kernel_bf16(
    __nv_bfloat16* data,
    long long N,
    cudaStream_t stream
) {
    if (N == 0) return;
    
    constexpr int BLOCK_SIZE = 256;

    // 基于向量数量和剩余元素数量计算Grid大小
    // grid-stride 循环可以优雅地处理任何数据量，不再需要min_blocks
    const int VEC_SIZE = 4;
    long long num_elements_to_process = (N / VEC_SIZE) + (N % VEC_SIZE);
    long long num_blocks = (num_elements_to_process + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    silu_inplace_kernel_bf16<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        data,
        N
    );
}


// SiLU in-place operation PyTorch binding
torch::Tensor silu_inplace_cuda(
    torch::Tensor input
) {
    TORCH_CHECK(input.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(input.is_contiguous(), "输入张量必须是连续的");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "输入张量必须是 BFloat16 类型");

    const long long N = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_silu_kernel_bf16(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        N,
        stream
    );

    // return input; // 返回原张量
}

// =================================================================
// 4. pybind11 模块定义
// =================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_argmax_softmax_cuda, "Fused Argmax-Softmax (CUDA, High-Performance, Unified)");
    m.def("silu_inplace", &silu_inplace_cuda, "In-place SiLU activation (BF16, Vectorized)");
}