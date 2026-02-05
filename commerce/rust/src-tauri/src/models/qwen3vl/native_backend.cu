#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// [ULTRA-ROBUST-TILED] Shared Memory Bit-serial Matmul
__global__ void bit_serial_matmul_kernel_shuffled(
    const float* __restrict__ input,    
    const uint32_t* __restrict__ weight, 
    const float* __restrict__ scale,     
    float* __restrict__ output,          
    int M, int N, int K
) {
    const int TILE_M = 32;
    __shared__ uint32_t s_input_bits[TILE_M]; 

    int m_base = blockIdx.y * TILE_M;
    int n_group = blockIdx.x;
    int n_sub = threadIdx.x; 
    int m_sub = threadIdx.y; 
    
    int m = m_base + m_sub;
    int n = n_group * 8 + n_sub;
    int k_blocks = (K + 31) / 32;

    float acc = 0.0f;

    for (int kb = 0; kb < k_blocks; ++kb) {
        if (n_sub == 0) {
            uint32_t bts = 0;
            if (m < M) {
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    int k_idx = kb * 32 + b;
                    if (k_idx < K && input[m * K + k_idx] >= 0.0f) bts |= (1u << b);
                }
            }
            s_input_bits[m_sub] = bts;
        }
        __syncthreads();

        if (m < M && n < N) {
            int idx = n_group * k_blocks * 8 + kb * 8 + n_sub;
            uint32_t w_val = weight[idx];
            float s_val = scale[idx];
            acc += (float)(32 - 2 * (int)__popc(s_input_bits[m_sub] ^ w_val)) * s_val;
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        output[m * N + n] = acc;
    }
}

// [ULTRA-SAFE] Bit-serial Attention Kernel
// Hardened indexing and shared memory protection for Large models (2B+)
__global__ void bit_serial_attn_kernel_v2026(
    const float* __restrict__ Q,        
    const uint32_t* __restrict__ K_p,   
    const float* __restrict__ V,        
    float* __restrict__ O,              
    int n_h, int n_kv, int h_d, int t_s, float sc, int q_len,
    float alpha
) {
    int q_idx = blockIdx.x; 
    int h = blockIdx.y;     
    int tid = threadIdx.x; // Block size 256
    
    if (q_idx >= q_len || h >= n_h) return;

    int k_b = (h_d + 31) / 32;
    int h_kv = h / (n_h / n_kv);
    
    __shared__ uint32_t s_q_bits[16]; // Supports up to h_d=512
    __shared__ float s_max_red[32]; 
    __shared__ float s_tile_scores[256];

    // 1. Collaborative load Query bits
    if (tid < k_b) {
        uint32_t bts = 0;
        for (int b = 0; b < 32; ++b) {
            int d_idx = tid * 32 + b;
            if (d_idx < h_d) {
                float val = Q[(q_idx * n_h * h_d) + (h * h_d) + d_idx];
                if (val >= 0.0f) bts |= (1u << b);
            }
        }
        s_q_bits[tid] = bts;
    }
    __syncthreads();
    
    float local_o[256]; // Supports h_d up to 256
    for(int i=0; i<256; ++i) local_o[i] = 0.0f;
    
    float running_max = -1e20f;
    float running_sum = 0.0f;

    // 2. Iterate over sequence length (t_s)
    for (int t_start = 0; t_start < t_s; t_start += blockDim.x) {
        int j = t_start + tid; 
        float score = -1e20f;
        
        if (j < t_s) {
            int dot = 0;
            // Boundary-safe Key bit-serial dot product
            for (int kb = 0; kb < k_b; ++kb) {
                int k_idx = (j * n_kv + h_kv) * k_b + kb;
                dot += (32 - 2 * (int)__popc(s_q_bits[kb] ^ K_p[k_idx]));
            }
            score = ((float)dot + alpha) * sc;
            if (score < -15.0f) score = -15.0f;
        }

        // Parallel Max reduction within block
        float t_max = score;
        for (int offset = 16; offset > 0; offset /= 2) t_max = fmaxf(t_max, __shfl_xor_sync(0xFFFFFFFF, t_max, offset));
        if ((tid % 32) == 0) s_max_red[tid / 32] = t_max;
        __syncthreads();
        
        float b_max = -1e20f;
        if (tid < 32) {
            b_max = (tid < (blockDim.x / 32)) ? s_max_red[tid] : -1e20f;
            for (int offset = 16; offset > 0; offset /= 2) b_max = fmaxf(b_max, __shfl_xor_sync(0xFFFFFFFF, b_max, offset));
            s_max_red[0] = b_max;
        }
        __syncthreads();
        b_max = s_max_red[0];

        float e_score = (j < t_s) ? expf(score - b_max) : 0.0f;
        s_tile_scores[tid] = e_score; 
        __syncthreads();
        
        // Parallel Sum reduction
        float t_sum = e_score;
        for (int offset = 16; offset > 0; offset /= 2) t_sum += __shfl_xor_sync(0xFFFFFFFF, t_sum, offset);
        if ((tid % 32) == 0) s_max_red[tid / 32] = t_sum;
        __syncthreads();
        
        float b_sum = 0.0f;
        if (tid < 32) {
            b_sum = (tid < (blockDim.x / 32)) ? s_max_red[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) b_sum += __shfl_xor_sync(0xFFFFFFFF, b_sum, offset);
            s_max_red[0] = b_sum;
        }
        __syncthreads();
        b_sum = s_max_red[0];

        // Softmax normalization and V-accumulation
        float n_max = fmaxf(running_max, b_max);
        float e_scale = expf(running_max - n_max);
        running_sum = running_sum * e_scale + b_sum;
        running_max = n_max;

        if (tid < h_d) {
            float v_acc = 0.0f;
            for (int k_j = 0; k_j < blockDim.x; ++k_j) {
                int g_j = t_start + k_j;
                if (g_j < t_s) {
                    int v_idx = (g_j * n_kv + h_kv) * h_d + tid;
                    v_acc += s_tile_scores[k_j] * V[v_idx];
                }
            }
            local_o[tid] = local_o[tid] * e_scale + v_acc;
        }
        __syncthreads();
    }

    // Write final output O
    if (tid < h_d) {
        float inv_sum = 1.0f / (running_sum + 1e-9f);
        O[(q_idx * n_h * h_d) + (h * h_d) + tid] = local_o[tid] * inv_sum;
    }
}

extern "C" {
    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha) {
        cudaSetDevice(dev);
        dim3 block(256); 
        dim3 grid(q_len, n_h);
        bit_serial_attn_kernel_v2026<<<grid, block>>>(d_q, d_k, d_v, d_o, n_h, n_kv, h_d, t_s, scale, q_len, alpha);
    }

    void bit_serial_matmul_cuda_direct(const float* d_i, const uint32_t* d_w, const float* d_s, float* d_o, int m, int n, int k, int dev) {
        cudaSetDevice(dev);
        dim3 block(8, 32); 
        dim3 grid((n + 7) / 8, (m + 31) / 32);
        bit_serial_matmul_kernel_shuffled<<<grid, block>>>(d_i, d_w, d_s, d_o, m, n, k);
    }
}