#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// --- 1-bit Matrix Multiplication Kernel ---
extern "C" __global__ void bit_serial_matmul_kernel(
    const float* input, const uint32_t* weight, const float* scales, float* output,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < M && col < N) {
        int k_blocks = K / 32;
        int total_dot = 0;
        for (int kb = 0; kb < k_blocks; ++kb) {
            uint32_t input_bits = 0;
            #pragma unroll
            for (int b = 0; b < 32; ++b) {
                if (input[row * K + kb * 32 + b] >= 0.0f) input_bits |= (1 << b);
            }
            total_dot += (32 - 2 * __popc(input_bits ^ weight[col * k_blocks + kb]));
        }
        output[row * N + col] = (float)total_dot * scales[col];
    }
}

// --- 1-bit KV Cache Attention Kernel (The "Perfect" Branch) ---
extern "C" __global__ void bit_serial_attn_kernel(
    const float* q,           // [n_heads, head_dim]
    const uint32_t* k_packed, // [total_seq_len, n_heads, k_blocks]
    const float* v,           // [total_seq_len, n_heads, head_dim]
    float* output,            // [n_heads, head_dim]
    int n_heads, int head_dim, int total_seq_len, float scale
) {
    int h = blockIdx.x; // Each block handles one head
    if (h >= n_heads) return;

    int k_blocks = head_dim / 32;
    extern __shared__ float s_scores[]; // Shared memory for attention scores

    // 1. Compute Attention Scores (Q_bits ^ K_bits)
    uint32_t q_packed[8]; // Max 256 head_dim
    for (int kb = 0; kb < k_blocks; kb++) {
        uint32_t bits = 0;
        for (int b = 0; b < 32; b++) {
            if (q[h * head_dim + kb * 32 + b] >= 0.0f) bits |= (1 << b);
        }
        q_packed[kb] = bits;
    }

    for (int t = threadIdx.x; t < total_seq_len; t += blockDim.x) {
        int dot = 0;
        for (int kb = 0; kb < k_blocks; kb++) {
            dot += (32 - 2 * __popc(q_packed[kb] ^ k_packed[(t * n_heads + h) * k_blocks + kb]));
        }
        s_scores[t] = (float)dot * scale;
    }
    __syncthreads();

    // 2. Softmax (Simple version for kernel)
    float max_s = -1e20f;
    for (int t = 0; t < total_seq_len; t++) if (s_scores[t] > max_s) max_s = s_scores[t];
    float sum_exp = 0.0f;
    for (int t = 0; t < total_seq_len; t++) {
        s_scores[t] = expf(s_scores[t] - max_s);
        sum_exp += s_scores[t];
    }
    float inv_sum = 1.0f / sum_exp;

    // 3. Weighted Sum (Score * V)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float res = 0.0f;
        for (int t = 0; t < total_seq_len; t++) {
            res += s_scores[t] * inv_sum * v[(t * n_heads + h) * head_dim + d];
        }
        output[h * head_dim + d] = res;
    }
}

extern "C" void bit_serial_matmul_cuda_direct(const float* d_i, const uint32_t* d_w, const float* d_s, float* d_o, int M, int N, int K, int dev) {
    cudaSetDevice(dev);
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    bit_serial_matmul_kernel<<<blocks, threads>>>(d_i, d_w, d_s, d_o, M, N, K);
}

extern "C" void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int h_d, int t_s, float scale, int dev) {
    cudaSetDevice(dev);
    // Launch with shared memory for scores
    bit_serial_attn_kernel<<<n_h, 256, t_s * sizeof(float)>>>(d_q, d_k, d_v, d_o, n_h, h_d, t_s, scale);
}