#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// [FINAL-SAFE-KERNEL] Guaranteed 1-bit Bit-serial Matmul
// Matches CPU logic 100% to ensure non-zero output.
__global__ void bit_serial_matmul_kernel_shuffled(
    const float* __restrict__ input,    
    const uint32_t* __restrict__ weight, 
    const float* __restrict__ scale,     
    float* __restrict__ output,          
    int M, int N, int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y; 
    int n = blockIdx.x * blockDim.x + threadIdx.x; 

    if (m >= M || n >= N) return;

    int k_blocks = (K + 31) / 32;
    int n_group = n / 8;
    int n_sub = n % 8;
    
    float acc = 0.0f;
    for (int kb = 0; kb < k_blocks; ++kb) {
        // 1. Pack input bits on the fly (Same as CPU)
        uint32_t input_bits = 0;
        for (int b = 0; b < 32; ++b) {
            int k_idx = kb * 32 + b;
            if (k_idx < K) {
                if (input[m * K + k_idx] >= 0.0f) input_bits |= (1u << b);
            }
        }
        
        // 2. Load weight and scale using Format 1 indexing
        int idx = n_group * k_blocks * 8 + kb * 8 + n_sub;
        uint32_t w_bits = weight[idx];
        float s_val = scale[idx];
        
        // 3. Bit-serial dot product
        uint32_t diff = input_bits ^ w_bits;
        int pop = __popc(diff);
        acc += (float)(32 - 2 * pop) * s_val;
    }
    
    output[m * N + n] = acc;
}

// [ULTRA-SAFE] Bit-serial Attention Kernel
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
    
    // Use smaller shared memory to avoid allocation issues
    __shared__ uint32_t s_q_bits[16]; 
    
    if (tid < k_b) {
        uint32_t bts = 0;
        for (int b = 0; b < 32; ++b) {
            int d_idx = tid * 32 + b;
            if (d_idx < h_d) {
                if (Q[(q_idx * n_h * h_d) + (h * h_d) + d_idx] >= 0.0f) bts |= (1u << b);
            }
        }
        s_q_bits[tid] = bts;
    }
    __syncthreads();
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    float local_o = 0.0f; // Each thread handles one dimension of O

    for (int j = 0; j < t_s; ++j) {
        // 1. Compute score (Bit-serial dot product)
        int dot = 0;
        for (int kb = 0; kb < k_b; ++kb) {
            dot += (32 - 2 * (int)__popc(s_q_bits[kb] ^ K_p[(j * n_kv + h_kv) * k_b + kb]));
        }
        float score = ((float)dot + alpha) * sc;
        if (score < -15.0f) score = -15.0f;

        // 2. Online Softmax update
        float n_max = fmaxf(running_max, score);
        float e_scale = expf(running_max - n_max);
        float e_score = expf(score - n_max);
        
        running_sum = running_sum * e_scale + e_score;
        running_max = n_max;

        // 3. Accumulate V
        if (tid < h_d) {
            local_o = local_o * e_scale + e_score * V[(j * n_kv + h_kv) * h_d + tid];
        }
    }

    // Write output
    if (tid < h_d) {
        O[(q_idx * n_h * h_d) + (h * h_d) + tid] = local_o / (running_sum + 1e-9f);
    }
}

extern "C" {
    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha) {
        cudaSetDevice(dev);
        dim3 block(h_d); // Each thread handles one HD dimension
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
