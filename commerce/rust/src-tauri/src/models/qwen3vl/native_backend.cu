#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// [OPTIMIZATION] Fast 1-bit Bit-serial Matmul Kernel
// Layout: Weights are shuffled into [N/8, K/32, 8]
__global__ void bit_serial_matmul_kernel_shuffled(
    const float* __restrict__ input,    // [M, K]
    const uint32_t* __restrict__ weight, // [N/8, K/32, 8] (Shuffled)
    const float* __restrict__ scale,     // [N]
    float* __restrict__ output,          // [M, N]
    int M, int N, int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y; // Row in M
    int n_group = blockIdx.x;                      // Group of 8 channels in N
    int n_sub = threadIdx.x;                       // 0~7 inside group
    int n = n_group * 8 + n_sub;                   // Actual channel index

    if (m >= M || n >= N) return;

    int k_blocks = (K + 31) / 32;
    float sum = 0.0f;

    // We process K in blocks of 32 bits
    for (int kb = 0; kb < k_blocks; ++kb) {
        // 1. Pack Input on the fly (or load pre-packed if optimized further)
        uint32_t input_bits = 0;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            int k_idx = kb * 32 + b;
            if (k_idx < K) {
                if (input[m * K + k_idx] >= 0.0f) {
                    input_bits |= (1u << b);
                }
            }
        }

        // 2. Load Shuffled Weight (Coalesced access for 8 threads in a group)
        // Weight Layout: [N/8][K_blocks][8]
        uint32_t weight_bits = weight[n_group * k_blocks * 8 + kb * 8 + n_sub];

        // 3. Bit-serial Dot Product (XNOR equivalent using XOR + Popcount)
        // Result = (32 - 2 * popcount(input ^ weight))
        uint32_t diff = __popc(input_bits ^ weight_bits);
        sum += (float)(32 - 2 * (int)diff);
    }

    // 4. Apply Scale and Store
    output[m * N + n] = sum * scale[n];
}

// [2026-OPTIMIZATION] Fused Bit-Flash Attention Kernel
// Uses Warp-Shuffle for zero-latency reduction and Shared-Memory for Q-bit caching
__global__ void bit_serial_attn_kernel_optimized(
    const float* __restrict__ Q,        
    const uint32_t* __restrict__ K_p,   
    const float* __restrict__ V,        
    float* __restrict__ O,              
    int n_h, int h_d, int t_s, float sc, int q_len
) {
    int q_idx = blockIdx.x; 
    int h = blockIdx.y;     
    int tid = threadIdx.x;
    if (q_idx >= q_len || h >= n_h) return;

    extern __shared__ float s_scores[]; 
    __shared__ uint32_t s_q_bits[8]; // Up to 256 dims

    int k_b = (h_d + 31) / 32;
    
    // 1. Parallel Q-Packing (Only once per block)
    if (tid < k_b) {
        uint32_t bts = 0;
        for (int b = 0; b < 32; ++b) {
            int dim_idx = tid * 32 + b;
            if (dim_idx < h_d) {
                if (Q[(q_idx * n_h * h_d) + (h * h_d) + dim_idx] >= 0.0f) bts |= (1u << b);
            }
        }
        s_q_bits[tid] = bts;
    }
    __syncthreads();

    // 2. Cooperative Score Calculation
    float thread_max = -1e20f;
    for (int j = tid; j < t_s; j += blockDim.x) {
        int dot = 0;
        #pragma unroll
        for (int kb = 0; kb < k_b; ++kb) {
            dot += (32 - 2 * __popc(s_q_bits[kb] ^ K_p[(j * n_h + h) * k_b + kb]));
        }
        float score = (float)dot * sc;
        s_scores[j] = score;
        thread_max = fmaxf(thread_max, score);
    }

    // 3. Warp-Level Max Reduction
    for (int offset = 16; offset > 0; offset /= 2)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    
    __shared__ float block_max;
    if (tid == 0) block_max = thread_max;
    __syncthreads();

    // 4. Parallel Softmax Exp and Sum
    float thread_sum = 0.0f;
    for (int j = tid; j < t_s; j += blockDim.x) {
        float e = expf(s_scores[j] - block_max);
        s_scores[j] = e;
        thread_sum += e;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    __shared__ float block_sum;
    if (tid == 0) block_sum = thread_sum + 1e-9f;
    __syncthreads();

    // 5. Weighted Sum (Score * V) with Coalesced memory access
    float inv_block_sum = 1.0f / block_sum;
    for (int d = tid; d < h_d; d += blockDim.x) {
        float out_val = 0.0f;
        for (int j = 0; j < t_s; ++j) {
            out_val += (s_scores[j] * inv_block_sum) * V[(j * n_h + h) * h_d + d];
        }
        O[(q_idx * n_h * h_d) + (h * h_d) + d] = out_val;
    }
}

extern "C" {
    void bit_serial_matmul_cuda_direct(const float* d_i, const uint32_t* d_w, const float* d_s, float* d_o, int m, int n, int k, int dev) {
        cudaSetDevice(dev);
        dim3 block(8, 16); 
        dim3 grid((n + 7) / 8, (m + 15) / 16);
        bit_serial_matmul_kernel_shuffled<<<grid, block>>>(d_i, d_w, d_s, d_o, m, n, k);
    }

    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int h_d, int t_s, float scale, int dev, int q_len) {
        cudaSetDevice(dev);
        dim3 block(256); 
        dim3 grid(q_len, n_h); 
        size_t shared_mem = t_s * sizeof(float);
        bit_serial_attn_kernel_optimized<<<grid, block, shared_mem>>>(d_q, (const uint32_t*)d_k, d_v, d_o, n_h, h_d, t_s, scale, q_len);
    }
}
