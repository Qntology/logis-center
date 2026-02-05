#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// [ULTRA-ROBUST-TILED-V2] Fixed Synchronization & Boundary Checks
__global__ void bit_serial_matmul_kernel_shuffled(
    const float* __restrict__ input,    
    const uint32_t* __restrict__ weight, 
    const float* __restrict__ scale,     
    float* __restrict__ output,          
    int M, int N, int K
) {
    const int TILE_M = 32;
    __shared__ uint32_t s_input_tile[TILE_M]; 

    int m_base = blockIdx.y * TILE_M;
    int n_group = blockIdx.x;
    int n_sub = threadIdx.x; 
    int m_sub = threadIdx.y; 
    
    int m = m_base + m_sub;
    int n = n_group * 8 + n_sub;
    int k_blocks = (K + 31) / 32;

    float acc = 0.0f;

    for (int kb = 0; kb < k_blocks; ++kb) {
        // 1. Collaborative load Input bits
        // EVERY thread in the block must reach __syncthreads()
        uint32_t bts = 0;
        if (n_sub == 0) {
            if (m < M) {
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    int k_idx = kb * 32 + b;
                    if (k_idx < K && input[m * K + k_idx] >= 0.0f) bts |= (1u << b);
                }
            }
            s_input_tile[m_sub] = bts;
        }
        
        // ALL threads must wait here, even those with m >= M
        __syncthreads();

        // 2. Compute
        if (m < M && n < N) {
            int idx = n_group * k_blocks * 8 + kb * 8 + n_sub;
            uint32_t w_val = weight[idx];
            float s_val = scale[idx];
            
            uint32_t diff = s_input_tile[m_sub] ^ w_val;
            acc += (float)(32 - 2 * (int)__popc(diff)) * s_val;
        }
        
        // Ensure shared memory is ready for next K-tile
        __syncthreads();
    }

    // 3. Final Store
    if (m < M && n < N) {
        output[m * N + n] = acc;
    }
}

// [FAST-ATTENTION] Optimized online softmax attention
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
    int tid = threadIdx.x;
    
    if (q_idx >= q_len || h >= n_h) return;

    int k_b = (h_d + 31) / 32;
    int h_kv = h / (n_h / n_kv);
    
    __shared__ uint32_t s_q_bits[16]; 
    
    if (tid < k_b) {
        uint32_t bts = 0;
        for (int b = 0; b < 32; ++b) {
            int d_idx = tid * 32 + b;
            if (d_idx < h_d && Q[(q_idx * n_h * h_d) + (h * h_d) + d_idx] >= 0.0f) bts |= (1u << b);
        }
        s_q_bits[tid] = bts;
    }
    __syncthreads();
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    float local_o = 0.0f; 

    for (int j = 0; j < t_s; ++j) {
        int dot = 0;
        for (int kb = 0; kb < k_b; ++kb) {
            dot += (32 - 2 * (int)__popc(s_q_bits[kb] ^ K_p[(j * n_kv + h_kv) * k_b + kb]));
        }
        float score = ((float)dot + alpha) * sc;
        if (score < -15.0f) score = -15.0f;

        float n_max = fmaxf(running_max, score);
        float e_scale = expf(running_max - n_max);
        float e_score = expf(score - n_max);
        
        running_sum = running_sum * e_scale + e_score;
        running_max = n_max;

        if (tid < h_d) {
            local_o = local_o * e_scale + e_score * V[(j * n_kv + h_kv) * h_d + tid];
        }
    }

    if (tid < h_d) {
        O[(q_idx * n_h * h_d) + (h * h_d) + tid] = local_o / (running_sum + 1e-9f);
    }
}

extern "C" {
    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha) {
        cudaSetDevice(dev);
        dim3 block(h_d); 
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
