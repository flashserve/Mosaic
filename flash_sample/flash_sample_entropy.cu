// file: flash_sample_entropy.cu
// Fused Argmax + Entropy kernel for entropy remasking strategy
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
__global__ void fused_argmax_entropy_kernel(
    const ScalarT* logits,
    long long* out_indices,
    float* out_entropy,
    long long M,
    long long N
) {
    long long row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= M) {
        return;
    }

    const ScalarT* row_logits = logits + row_idx * N;
    
    // 步骤 1: 线程级归约 - 找最大值和 argmax
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

            if (other_val > s_vals[tid]) {
                s_vals[tid] = other_val;
                s_idxs[tid] = other_idx;
            } else if (other_val == s_vals[tid]) {
                // 如果值相等，选择索引更小的那个，以匹配 PyTorch 的行为
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
    
    float final_denominator = s_vals[0];
    __syncthreads();
    
    // 步骤 4: 计算 Entropy = sum(p * log(p))
    // log(sum(exp(x - max))) 用于后续计算
    float log_sum = logf(final_denominator);
    
    float thread_entropy = 0.0f;
    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float logit = scalar_to_float(row_logits[col_idx]);
        // p = exp(logit - max) / sum
        float log_p = logit - final_max_val - log_sum;  // log(p)
        float p = expf(log_p);                          // p
        // entropy += p * log(p)
        thread_entropy += p * log_p;
    }
    
    s_vals[tid] = thread_entropy;
    __syncthreads();
    
    // 步骤 5: 块内归约 entropy
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            s_vals[tid] += s_vals[tid + offset];
        }
        __syncthreads();
    }

    // 步骤 6: 线程 0 将最终结果写回全局显存
    if (tid == 0) {
        float final_entropy = s_vals[0];
        out_entropy[row_idx] = final_entropy;  // 负熵（因为 log(p) < 0）
        out_indices[row_idx] = final_argmax_idx;
    }
}


// =================================================================
// 2. C++ 启动器部分
// =================================================================
template <typename ScalarT>
void launch_fused_entropy_kernel_typed(
    const ScalarT* logits,
    long long* out_indices,
    float* out_entropy,
    long long M,
    long long N,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 1024;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    
    fused_argmax_entropy_kernel<BLOCK_SIZE, ScalarT><<<grid, block, 0, stream>>>(
        logits,
        out_indices,
        out_entropy,
        M,
        N
    );
}


// =================================================================
// 3. PyTorch 绑定部分
// =================================================================
std::vector<torch::Tensor> entropy_cuda(
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
    auto out_entropy = torch::empty({M}, logits.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (logits.scalar_type() == torch::kFloat16) {
        launch_fused_entropy_kernel_typed<half>(
            reinterpret_cast<const half*>(logits.data_ptr()),
            (long long*)out_indices.data_ptr(),
            (float*)out_entropy.data_ptr(),
            M,
            N,
            stream
        );
    } else {
        launch_fused_entropy_kernel_typed<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            (long long*)out_indices.data_ptr(),
            (float*)out_entropy.data_ptr(),
            M,
            N,
            stream
        );
    }
    
    return {out_indices, out_entropy};
}


// =================================================================
// 4. pybind11 模块定义
// =================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("entropy", &entropy_cuda, "Fused Argmax-Entropy for entropy remasking (CUDA)");
}

