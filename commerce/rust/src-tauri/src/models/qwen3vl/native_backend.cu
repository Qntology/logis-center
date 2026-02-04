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

// [OPTIMIZATION] Bit-serial Attention Kernel (Simplified for 1-bit)
__global__ void bit_serial_attn_kernel_shuffled(
    const float* __restrict__ Q,        // [seq_len, head_dim]
    const uint32_t* __restrict__ K_p,   // [total_kv_len, head_dim/32] (Packed)
    const float* __restrict__ V,        // [total_kv_len, head_dim]
    float* __restrict__ O,              // [seq_len, head_dim]
    int n_h, int h_d, int t_s, float sc
) {
    int q_idx = blockIdx.x; // Current token being generated
    int h = blockIdx.y;     // Head index
    if (q_idx >= 1 || h >= n_h) return; // Simplified for single token generation

    extern __shared__ float s_scores[]; // Shared memory for softmax scores [t_s]

    int k_b = (h_d + 31) / 32;
    
    // 1. Compute Dot Product (Q * K)
    for (int j = threadIdx.x; j < t_s; j += blockDim.x) {
        uint32_t q_bits[8]; // Max head_dim 256
        #pragma unroll
        for (int kb = 0; kb < k_b; ++kb) {
            uint32_t bts = 0;
            for (int b = 0; b < 32; ++b) {
                if (Q[(h * h_d) + kb * 32 + b] >= 0.0f) bts |= (1u << b);
            }
            q_bits[kb] = bts;
        }

        int dot = 0;
        for (int kb = 0; kb < k_b; ++kb) {
            uint32_t kj = K_p[(j * n_h + h) * k_b + kb];
            dot += (32 - 2 * __popc(q_bits[kb] ^ kj));
        }
        s_scores[j] = (float)dot * sc;
    }
    __syncthreads();

    // 2. Softmax (Simple version for brevity)
    // ... (Softmax logic here)
    
    // 3. Weighted Sum (Score * V)
    // ... (V accumulation logic here)
}

extern "C" {
    void bit_serial_matmul_cuda_direct(const float* d_i, const uint32_t* d_w, const float* d_s, float* d_o, int m, int n, int k, int dev) {
        cudaSetDevice(dev);
        
        dim3 block(8, 16); // 8 threads for N-group, 16 for M rows
        dim3 grid((n + 7) / 8, (m + 15) / 16);
        
        bit_serial_matmul_kernel_shuffled<<<grid, block>>>(d_i, d_w, d_s, d_o, m, n, k);
    }

    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int h_d, int t_s, float scale, int dev) {
        cudaSetDevice(dev);
        // Optimized attention dispatch...
    }
}
