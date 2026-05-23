// file: flash_sample_entropy_witht.cu
// Fused Gumbel-Max + Entropy kernel with temperature support
#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <float.h>
#include <chrono>

// =================================================================
// 1. CUDA Kernel 部分
// =================================================================
// 设备端标量到 float 的转换重载
__device__ __forceinline__ float scalar_to_float(half v) {
    return __half2float(v);
}

__device__ __forceinline__ float scalar_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <int BLOCK_SIZE, typename ScalarT>
__global__ void fused_gumbel_max_entropy_kernel(
    const ScalarT* logits,
    long long* out_indices,
    float* out_entropy,
    long long M,
    long long N,
    float temperature,
    unsigned long long seed
) {
    long long row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= M) {
        return;
    }

    const ScalarT* row_logits = logits + row_idx * N;
    
    // 初始化每个线程的 RNG 状态
    curandState local_state;
    curand_init(seed, row_idx * BLOCK_SIZE + tid, 0, &local_state);
    
    // 步骤 1: 线程级归约 - 找 logits/T + Gumbel 噪声的最大值和 argmax
    float thread_max_val = -FLT_MAX;
    long long thread_argmax_idx = -1;
    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float logit = scalar_to_float(row_logits[col_idx]);
        
        // Temperature 缩放
        logit *= inv_temp;
        
        // 生成 Gumbel 噪声: -log(-log(U))
        float u = curand_uniform(&local_state);
        u = fmaxf(u, 1e-10f);  // 避免 log(0)
        float gumbel = -logf(-logf(u));
        
        float val_with_noise = logit + gumbel;
        
        if (val_with_noise > thread_max_val) {
            thread_max_val = val_with_noise;
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
    
    long long final_argmax_idx = s_idxs[BLOCK_SIZE - 1];
    
    // 步骤 3: 计算原始 logits（缩放后）的最大值，用于数值稳定的 softmax
    float thread_logit_max = -FLT_MAX;
    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float logit = scalar_to_float(row_logits[col_idx]) * inv_temp;
        thread_logit_max = fmaxf(thread_logit_max, logit);
    }
    
    s_vals[tid] = thread_logit_max;
    __syncthreads();
    
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            s_vals[tid] = fmaxf(s_vals[tid], s_vals[tid + offset]);
        }
        __syncthreads();
    }
    
    float final_logit_max = s_vals[0];
    __syncthreads();
    
    // 步骤 4: 计算 Softmax 分母
    float thread_denominator = 0.0f;
    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float logit = scalar_to_float(row_logits[col_idx]) * inv_temp;
        thread_denominator += expf(logit - final_logit_max);
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
    
    // 步骤 5: 计算 Entropy = sum(p * log(p))
    float log_sum = logf(final_denominator);
    
    float thread_entropy = 0.0f;
    for (long long col_idx = tid; col_idx < N; col_idx += BLOCK_SIZE) {
        float logit = scalar_to_float(row_logits[col_idx]) * inv_temp;
        float log_p = logit - final_logit_max - log_sum;
        float p = expf(log_p);
        thread_entropy += p * log_p;
    }
    
    s_vals[tid] = thread_entropy;
    __syncthreads();
    
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            s_vals[tid] += s_vals[tid + offset];
        }
        __syncthreads();
    }

    // 步骤 6: 线程 0 将最终结果写回全局显存
    if (tid == 0) {
        float final_entropy = s_vals[0];
        out_entropy[row_idx] = final_entropy;
        out_indices[row_idx] = final_argmax_idx;
    }
}


// =================================================================
// 2. C++ 启动器部分
// =================================================================
template <typename ScalarT>
void launch_fused_gumbel_max_entropy_kernel_typed(
    const ScalarT* logits,
    long long* out_indices,
    float* out_entropy,
    long long M,
    long long N,
    float temperature,
    unsigned long long seed,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 1024;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    
    fused_gumbel_max_entropy_kernel<BLOCK_SIZE, ScalarT><<<grid, block, 0, stream>>>(
        logits,
        out_indices,
        out_entropy,
        M,
        N,
        temperature,
        seed
    );
}


// =================================================================
// 3. PyTorch 绑定部分
// =================================================================
std::vector<torch::Tensor> entropy_witht_cuda(
    torch::Tensor logits,
    float temperature
) {
    TORCH_CHECK(logits.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(logits.dim() == 2, "输入张量必须是 2D");
    TORCH_CHECK(logits.is_contiguous(), "输入张量必须是连续的");
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat16 || logits.scalar_type() == torch::kBFloat16,
        "输入张量必须是 Float16 或 BFloat16 类型"
    );
    TORCH_CHECK(temperature >= 0.0f, "Temperature 必须 >= 0");

    const long long M = logits.size(0);
    const long long N = logits.size(1);

    auto out_indices = torch::empty({M}, logits.options().dtype(torch::kInt64));
    auto out_entropy = torch::empty({M}, logits.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // 生成随机种子（使用时间戳）
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    if (logits.scalar_type() == torch::kFloat16) {
        launch_fused_gumbel_max_entropy_kernel_typed<half>(
            reinterpret_cast<const half*>(logits.data_ptr()),
            (long long*)out_indices.data_ptr(),
            (float*)out_entropy.data_ptr(),
            M,
            N,
            temperature,
            seed,
            stream
        );
    } else {
        launch_fused_gumbel_max_entropy_kernel_typed<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            (long long*)out_indices.data_ptr(),
            (float*)out_entropy.data_ptr(),
            M,
            N,
            temperature,
            seed,
            stream
        );
    }
    
    return {out_indices, out_entropy};
}


// =================================================================
// 4. pybind11 模块定义
// =================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("entropy_witht", &entropy_witht_cuda, "Fused Gumbel-Max-Entropy with temperature (CUDA)");
}

