#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

// [2026-SPEED-MASTER] Ultra-Fast Tiled Bit-serial Matmul for F16
__global__ void bit_serial_matmul_kernel_f16(
    const half* __restrict__ input,    
    const uint32_t* __restrict__ weight, 
    const half* __restrict__ scale,     
    half* __restrict__ output,          
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
                    if (k_idx < K) {
                        if (__half2float(input[m * K + k_idx]) >= 0.0f) bts |= (1u << b);
                    }
                }
            }
            s_input_bits[m_sub] = bts;
        }
        __syncthreads();

        if (m < M && n < N) {
            int idx = n_group * k_blocks * 8 + kb * 8 + n_sub;
            uint32_t w_val = weight[idx];
            float s_val = __half2float(scale[idx]);
            uint32_t diff = s_input_bits[m_sub] ^ w_val;
            acc += (float)(32 - 2 * (int)__popc(diff)) * s_val;
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        output[m * N + n] = __float2half(acc);
    }
}

// [FAST-ATTENTION] High-speed online softmax attention for F16
__global__ void bit_serial_attn_kernel_f16(
    const half* __restrict__ Q,        
    const uint32_t* __restrict__ K_p,   
    const half* __restrict__ V,        
    half* __restrict__ O,              
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
            if (d_idx < h_d && __half2float(Q[(q_idx * n_h * h_d) + (h * h_d) + d_idx]) >= 0.0f) bts |= (1u << b);
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
            local_o = local_o * e_scale + e_score * __half2float(V[(j * n_kv + h_kv) * h_d + tid]);
        }
    }

    if (tid < h_d) {
        O[(q_idx * n_h * h_d) + (h * h_d) + tid] = __float2half(local_o / (running_sum + 1e-9f));
    }
}

// [LEGACY] Kernels for Float32 support
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
            acc += (float)(32 - 2 * (int)__popc(s_input_bits[m_sub] ^ weight[idx])) * scale[idx];
        }
        __syncthreads();
    }
    if (m < M && n < N) output[m * N + n] = acc;
}

__global__ void bit_serial_attn_kernel_v2026(
    const float* __restrict__ Q,        
    const uint32_t* __restrict__ K_p,   
    const float* __restrict__ V,        
    float* __restrict__ O,              
    int n_h, int n_kv, int h_d, int t_s, float sc, int q_len,
    float alpha
) {
    int q_idx = blockIdx.x; int h = blockIdx.y; int tid = threadIdx.x;
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
    float running_max = -1e20f; float running_sum = 0.0f; float local_o = 0.0f; 
    for (int j = 0; j < t_s; ++j) {
        int dot = 0;
        for (int kb = 0; kb < k_b; ++kb) dot += (32 - 2 * (int)__popc(s_q_bits[kb] ^ K_p[(j * n_kv + h_kv) * k_b + kb]));
        float score = ((float)dot + alpha) * sc;
        if (score < -15.0f) score = -15.0f;
        float n_max = fmaxf(running_max, score);
        float e_scale = expf(running_max - n_max);
        float e_score = expf(score - n_max);
        running_sum = running_sum * e_scale + e_score;
        running_max = n_max;
        if (tid < h_d) local_o = local_o * e_scale + e_score * V[(j * n_kv + h_kv) * h_d + tid];
    }
    if (tid < h_d) O[(q_idx * n_h * h_d) + (h * h_d) + tid] = local_o / (running_sum + 1e-9f);
}

// --- [CRITICAL] EXTERN C WRAPPERS ---

extern "C" {

    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha) {

        bit_serial_attn_kernel_v2026<<<dim3(q_len, n_h), h_d>>>(d_q, d_k, d_v, d_o, n_h, n_kv, h_d, t_s, scale, q_len, alpha);

    }



    void bit_serial_attn_cuda_f16(const half* d_q, const uint32_t* d_k, const half* d_v, half* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha) {

        bit_serial_attn_kernel_f16<<<dim3(q_len, n_h), h_d>>>(d_q, d_k, d_v, d_o, n_h, n_kv, h_d, t_s, scale, q_len, alpha);

    }



    void bit_serial_matmul_cuda_direct(const float* d_i, const uint32_t* d_w, const float* d_s, float* d_o, int m, int n, int k, int dev) {

        bit_serial_matmul_kernel_shuffled<<<dim3((n + 7) / 8, (m + 31) / 32), dim3(8, 32)>>>(d_i, d_w, d_s, d_o, m, n, k);

    }



        void bit_serial_matmul_cuda_f16(const half* d_i, const uint32_t* d_w, const half* d_s, half* d_o, int m, int n, int k, int dev) {



            bit_serial_matmul_kernel_f16<<<dim3((n + 7) / 8, (m + 31) / 32), dim3(8, 32)>>>(d_i, d_w, d_s, d_o, m, n, k);



        }



    }



    



    // [STANDARD-F16-MATMUL] For non-quantized layers like lm_head or standard projections



    __global__ void standard_matmul_kernel_f16(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {



        int m = blockIdx.y * blockDim.y + threadIdx.y;



        int n = blockIdx.x * blockDim.x + threadIdx.x;



        if (m < M && n < N) {



            float acc = 0.0f;



            for (int k = 0; k < K; ++k) {



                acc += __half2float(A[m * K + k]) * __half2float(B[n * K + k]); // Assumes B is [N, K] and transposed or handled accordingly



            }



            C[m * N + n] = __float2half(acc);



        }



    }



    



    extern "C" {



    



        void standard_matmul_cuda_f16(const half* d_i, const half* d_w, half* d_o, int m, int n, int k) {



    



            dim3 block(16, 16);



    



            dim3 grid((n + 15) / 16, (m + 15) / 16);



    



            standard_matmul_kernel_f16<<<grid, block>>>(d_i, d_w, d_o, m, n, k);



    



        }



    



    



    



            void rms_norm_cuda_f16(const half* d_i, const half* d_w, half* d_o, int m, int hid, float eps) {



    



    



    



                // [FIX] Call the standalone kernel instead of using an invalid inline lambda



    



    



    



                rms_norm_kernel_f16<<<m, 256, 256 * sizeof(float)>>>(d_i, d_w, d_o, hid, eps);



    



    



    



            }



    



    



    



        }



    



    



    



        



    



    



    



    // Actual Kernels for activation and norm



    



    __global__ void rms_norm_kernel_f16(const half* i, const half* w, half* o, int h, float e) {



    



        int row = blockIdx.x;



    



        int tid = threadIdx.x;



    



        extern __shared__ float s_part_sum[];



    



        float sum = 0.0f;



    



        for (int j = tid; j < h; j += blockDim.x) { float val = __half2float(i[row * h + j]); sum += val * val; }



    



        s_part_sum[tid] = sum;



    



        __syncthreads();



    



        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {



    



            if (tid < stride) s_part_sum[tid] += s_part_sum[tid + stride];



    



            __syncthreads();



    



        }



    



        float inv_rms = rsqrtf(s_part_sum[0] / h + e);



    



        for (int j = tid; j < h; j += blockDim.x) o[row * h + j] = __float2half(__half2float(i[row * h + j]) * inv_rms * __half2float(w[j]));



    



    }



    



    



    



    __global__ void silu_mul_kernel_f16(half* gate, const half* up, int size) {



    



        int idx = blockIdx.x * blockDim.x + threadIdx.x;



    



        if (idx < size) {



    



            float g = __half2float(gate[idx]);



    



            float u = __half2float(up[idx]);



    



            gate[idx] = __float2half((g / (1.0f + expf(-g))) * u);



    



        }



    



    }



    



    



    



    extern "C" {



    



        void cuda_rms_norm_f16(const half* d_i, const half* d_w, half* d_o, int m, int hid, float eps) {



    



            rms_norm_kernel_f16<<<m, 256, 256 * sizeof(float)>>>(d_i, d_w, d_o, hid, eps);



    



        }



    



        void cuda_silu_mul_f16(half* d_gate, const half* d_up, int size) {



    



            silu_mul_kernel_f16<<<(size + 255) / 256, 256>>>(d_gate, d_up, size);



    



        }



    



    }



    



    



    
