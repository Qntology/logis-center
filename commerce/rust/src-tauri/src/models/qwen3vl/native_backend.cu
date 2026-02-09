#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

// ----------------------------------------------------------------------------
// 1. CUDA Kernels (Pure C++ context)
// ----------------------------------------------------------------------------

// [2026-PRECISION-MATCHING] Use float* for scales to support 10^-9 precision
__global__ void bit_serial_matmul_kernel_f16(const half* input, const uint32_t* weight, const float* scale, half* output, int M, int N, int K, int src_k) {
    const int TILE_M = 32;
    __shared__ uint32_t s_input_bits[TILE_M]; 
    __shared__ float s_input_scales[TILE_M];
    
    int m = blockIdx.y * TILE_M + threadIdx.y;
    int n = blockIdx.x * 8 + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    int src_k_blocks = (src_k + 31) / 32;
    int K_blocks = (K + 31) / 32;
    int repetition = (K > src_k) ? (K / src_k) : 1;
    float r_scale = rsqrtf((float)repetition);
    float acc = 0.0f;
    
    for (int kb = 0; kb < src_k_blocks; ++kb) {
        if (threadIdx.x == 0) {
            uint32_t folded_bts = 0; 
            float block_sum_sq = 0.0f;
            for (int b = 0; b < 32; ++b) {
                int k_base = kb * 32 + b;
                if (k_base < src_k) {
                    float sig = 0.0f;
                    for (int r = 0; r < repetition; ++r) {
                        sig += __half2float(input[m * K + (k_base + r * src_k) % K]);
                    }
                    sig *= r_scale;
                    if (sig >= 0.0f) folded_bts |= (1u << b);
                    block_sum_sq += sig * sig;
                }
            }
            s_input_bits[threadIdx.y] = folded_bts;
            s_input_scales[threadIdx.y] = sqrtf(block_sum_sq / 32.0f + 1e-8f);
        }
        __syncthreads();
        
        int idx = blockIdx.x * K_blocks * 8 + kb * 8 + threadIdx.x;
        // [MATCHING] Higher-precision scale application
        float w_scale = scale[idx]; 
        acc += (float)(32 - 2 * (int)__popc(s_input_bits[threadIdx.y] ^ weight[idx])) * w_scale * s_input_scales[threadIdx.y];
        __syncthreads();
    }
    output[m * N + n] = __float2half(acc);
}

__global__ void bit_serial_matmul_kernel_4bit_f16(const half* input, const uint32_t* weight, const float* scale, half* output, int M, int N, int K, int src_k) {
    const int TILE_M = 32;
    __shared__ uint32_t s_input_bits[TILE_M]; 
    __shared__ float s_input_scales[TILE_M];
    
    int m = blockIdx.y * TILE_M + threadIdx.y;
    int n = blockIdx.x * 8 + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    int src_k_blocks = (src_k + 31) / 32;
    int K_blocks = (K + 31) / 32;
    int slice_stride = ((N + 7) / 8) * K_blocks * 8;
    float acc = 0.0f;
    
    for (int kb = 0; kb < src_k_blocks; ++kb) {
        if (threadIdx.x == 0) {
            uint32_t folded_bts = 0; 
            float block_sum_sq = 0.0f;
            for (int b = 0; b < 32; ++b) {
                int k_idx = kb * 32 + b;
                if (k_idx < src_k) {
                    float sig = __half2float(input[m * K + k_idx % K]);
                    if (sig >= 0.0f) folded_bts |= (1u << b);
                    block_sum_sq += sig * sig;
                }
            }
            s_input_bits[threadIdx.y] = folded_bts;
            s_input_scales[threadIdx.y] = sqrtf(block_sum_sq / 32.0f + 1e-8f);
        }
        __syncthreads();
        
        int base_idx = blockIdx.x * K_blocks * 8 + kb * 8 + threadIdx.x;
        float b_acc = 0.0f;
        for (int s = 0; s < 4; ++s) {
            b_acc += (float)(32 - 2 * (int)__popc(s_input_bits[threadIdx.y] ^ weight[s * slice_stride + base_idx])) * (float)(1 << s);
        }
        // [MATCHING] Use F32 scale for 10^-9 precision matching
        float w_scale = scale[base_idx];
        acc += (b_acc - 8.0f * 32.0f) * w_scale * s_input_scales[threadIdx.y];
        __syncthreads();
    }
    output[m * N + n] = __float2half(acc);
}

__global__ void bit_serial_attn_kernel_f16(const half* Q, const uint32_t* K_p, const half* V, half* O, int n_h, int n_kv, int h_d, int t_s, float sc, int q_l, float alpha, int src_h_d) {
    int h = blockIdx.x; 
    int q_idx = blockIdx.y; 
    int tid = threadIdx.x;
    
    if (h >= n_h || q_idx >= q_l) return;
    
    int h_kv = h / (n_h / n_kv); 
    int h_d_blocks = (h_d + 31) / 32; 
    int src_kb = (src_h_d + 31) / 32;
    
    extern __shared__ uint32_t s_q_bits[]; 
    float* s_q_scales = (float*)&s_q_bits[src_kb];
    
    if (tid < src_h_d) {
        float q_val = __half2float(Q[q_idx * n_h * h_d + h * h_d + tid]);
        int b_idx = tid / 32; 
        int bit = tid % 32;
        atomicOr(&s_q_bits[b_idx], (q_val >= 0.0f ? (1u << bit) : 0u));
        atomicAdd(&s_q_scales[b_idx], q_val * q_val);
    }
    __syncthreads();
    
    if (tid < src_kb && tid == (tid / 32) * 32) {
        s_q_scales[tid / 32] = sqrtf(s_q_scales[tid / 32] / 32.0f + 1e-8f);
    }
    __syncthreads();
    
    float r_max = -10000.0f; 
    float r_sum = 0.0f; 
    float l_o = 0.0f;
    
    for (int j = 0; j < t_s; ++j) {
        float dot = 0.0f;
        for (int kb = 0; kb < src_kb; ++kb) {
            dot += (float)(32 - 2 * (int)__popc(s_q_bits[kb] ^ K_p[(j * n_kv + h_kv) * h_d_blocks + kb])) * s_q_scales[kb];
        }
        float score = fmaxf(-20.0f, fminf(20.0f, (dot + alpha) * sc));
        float n_max = fmaxf(r_max, score); 
        float e_scale = expf(r_max - n_max); 
        float e_score = expf(score - n_max);
        r_sum = r_sum * e_scale + e_score; 
        r_max = n_max;
        if (tid < h_d) {
            l_o = l_o * e_scale + e_score * __half2float(V[(j * n_kv + h_kv) * h_d + tid]);
        }
    }
    if (tid < h_d) {
        O[(q_idx * n_h * h_d) + (h * h_d) + tid] = __float2half(l_o / (r_sum + 1e-12f));
    }
}

__global__ void standard_matmul_kernel_f16(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.y * 16 + threadIdx.y; 
    int n = blockIdx.x * 16 + threadIdx.x;
    if (m < M && n < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
        C[m * N + n] = __float2half(acc);
    }
}

__global__ void rms_norm_kernel_f16(const half* i, const half* w, half* o, int h, float e) {
    int row = blockIdx.x; 
    int tid = threadIdx.x; 
    float sum = 0.0f;
    for (int col = tid; col < h; col += blockDim.x) { 
        float v = __half2float(i[row * h + col]); 
        sum += v * v; 
    }
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    __shared__ float s_warp_sums[32]; 
    if (tid % 32 == 0) s_warp_sums[tid / 32] = sum; 
    __syncthreads();
    if (tid < 32) { 
        float s = (tid < (blockDim.x + 31) / 32) ? s_warp_sums[tid] : 0.0f; 
        for (int offset = 16; offset > 0; offset /= 2) s += __shfl_down_sync(0xFFFFFFFF, s, offset); 
        if (tid == 0) s_warp_sums[0] = s; 
    }
    __syncthreads();
    float inv_rms = rsqrtf(s_warp_sums[0] / h + e);
    for (int col = tid; col < h; col += blockDim.x) o[row * h + col] = __float2half(__half2float(i[row * h + col]) * inv_rms * __half2float(w[col]));
}

__global__ void silu_mul_kernel_f16(half* gate, const half* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        float g = __half2float(gate[idx]); 
        gate[idx] = __float2half((g / (1.0f + expf(-g))) * __half2float(up[idx])); 
    }
}

__global__ void apply_rope_inplace_kernel_f16(half* Q, half* K, const half* cos_table, const half* sin_table, int q_len, int s_o, int n_h, int n_kv, int h_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int head_idx = idx / h_d; 
    int d_idx = idx % h_d;
    if (d_idx >= h_d / 2) return;
    int token_idx = head_idx / (n_h + n_kv); 
    int p = token_idx + s_o; 
    int target_head = head_idx % (n_h + n_kv);
    half* ptr = (target_head < n_h) ? &Q[token_idx * n_h * h_d + target_head * h_d] : &K[token_idx * n_kv * h_d + (target_head - n_h) * h_d];
    float q0 = __half2float(ptr[d_idx]); 
    float q1 = __half2float(ptr[d_idx + h_d / 2]);
    float c = __half2float(cos_table[p * (h_d / 2) + d_idx]); 
    float s = __half2float(sin_table[p * (h_d / 2) + d_idx]);
    ptr[d_idx] = __float2half(q0 * c - q1 * s); 
    ptr[d_idx + h_d / 2] = __float2half(q0 * s + q1 * c);
}

__global__ void pack_f16_to_u32_kernel(const half* src, uint32_t* dst, int elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid * 32 >= elements) return;
    uint32_t p = 0; 
    for (int i = 0; i < 32; ++i) { 
        int idx = tid * 32 + i; 
        if (idx < elements && __half2float(src[idx]) >= 0.0f) p |= (1u << i); 
    }
    dst[tid] = p;
}

__global__ void apply_gain_kernel_f16(half* data, float gain, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < total) data[i] = __float2half(__half2float(data[i]) * gain);
}

__global__ void add_inplace_kernel_f16(half* dst, const half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size) dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

__global__ void hybrid_repeat_kernel_f16(half* data, int src_size, int target_size, int q_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= q_len * target_size) return;
    int i = tid % target_size; 
    if (i >= src_size) data[tid] = data[(tid / target_size) * target_size + (i % src_size)];
}

__global__ void silu_inplace_kernel_f16(half* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size) { 
        float f = __half2float(data[i]); 
        data[i] = __float2half(f / (1.0f + expf(-f))); 
    }
}

__global__ void element_mul_kernel_f16(half* dst, const half* src, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < total) dst[i] = __float2half(__half2float(dst[i]) * __half2float(src[i]));
}

__global__ void f32_matmul_bias_sharpen_kernel(const float* A, const float* B, const float* Bias, float* C, int M, int N, int K, float sharpen) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f; 
        for (int i = 0; i < K; ++i) acc += A[row * K + i] * B[col * K + i];
        if (Bias) acc += Bias[col]; 
        C[row * N + col] = acc * sharpen;
    }
}

// ----------------------------------------------------------------------------
// 2. Extern C Wrappers (Safe for MSVC linking)
// ----------------------------------------------------------------------------

extern "C" {
    void bit_serial_matmul_cuda_f16(const half* d_i, const uint32_t* d_w, const float* d_s, half* d_o, int m, int n, int k, int dev, int src_k) {
        bit_serial_matmul_kernel_f16<<<dim3((n+7)/8, (m+31)/32), dim3(8, 32)>>>(d_i, d_w, d_s, d_o, m, n, k, src_k);
    }
    void bit_serial_matmul_cuda_4bit_f16(const half* d_i, const uint32_t* d_w, const float* d_s, half* d_o, int m, int n, int k, int dev, int src_k) {
        bit_serial_matmul_kernel_4bit_f16<<<dim3((n+7)/8, (m+31)/32), dim3(8, 32)>>>(d_i, d_w, d_s, d_o, m, n, k, src_k);
    }
    void bit_serial_attn_cuda_f16(const half* d_q, const uint32_t* d_k, const half* d_v, half* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha, int src_h_d) {
        bit_serial_attn_kernel_f16<<<dim3(n_h, q_len), dim3(h_d < 1024 ? h_d : 1024), (src_h_d+31)/32 * (4+4)>>>(d_q, d_k, d_v, d_o, n_h, n_kv, h_d, t_s, scale, q_len, alpha, src_h_d);
    }
    void standard_matmul_cuda_f16(const half* d_i, const half* d_w, half* d_o, int m, int n, int k) {
        standard_matmul_kernel_f16<<<dim3((n+15)/16, (m+15)/16), dim3(16, 16)>>>(d_i, d_w, d_o, m, n, k);
    }
    void cuda_rms_norm_f16(const half* d_i, const half* d_w, half* d_o, int m, int hid, float eps) {
        rms_norm_kernel_f16<<<m, 256>>>(d_i, d_w, d_o, hid, eps);
    }
    void cuda_silu_mul_f16(half* d_gate, const half* d_up, int size) {
        silu_mul_kernel_f16<<<(size+255)/256, 256>>>(d_gate, d_up, size);
    }
    void cuda_apply_rope_f16(half* d_q, half* d_k, const half* d_cos, const half* d_sin, int q_len, int s_o, int n_h, int n_kv, int h_d) {
        apply_rope_inplace_kernel_f16<<<((n_h+n_kv)*q_len*(h_d/2)+255)/256, 256>>>(d_q, d_k, d_cos, d_sin, q_len, s_o, n_h, n_kv, h_d);
    }
    void cuda_pack_bits_f16(const half* d_src, uint32_t* d_dst, int elements) {
        pack_f16_to_u32_kernel<<<((elements+31)/32+255)/256, 256>>>(d_src, d_dst, elements);
    }
    void cuda_add_inplace_f16(half* d_dst, const half* d_src, int size) {
        add_inplace_kernel_f16<<<(size+255)/256, 256>>>(d_dst, d_src, size);
    }
    void cuda_hybrid_repeat_f16(half* d_data, int src_size, int target_size, int q_len) {
        hybrid_repeat_kernel_f16<<<(q_len*target_size+255)/256, 256>>>(d_data, src_size, target_size, q_len);
    }
    void cuda_silu_inplace_f16(half* d_data, int size) {
        silu_inplace_kernel_f16<<<(size+255)/256, 256>>>(d_data, size);
    }
    void cuda_apply_gain_f16(half* d_data, float gain, int elements) {
        apply_gain_kernel_f16<<<(elements+255)/256, 256>>>(d_data, gain, elements);
    }
    void cuda_element_mul_f16(half* d_dst, const half* d_src, int size) {
        element_mul_kernel_f16<<<(size+255)/256, 256>>>(d_dst, d_src, size);
    }
    void high_precision_matmul_f32_bias(const float* d_i, const float* d_w, const float* d_b, float* d_o, int m, int n, int k, float sharpen) {
        f32_matmul_bias_sharpen_kernel<<<dim3((n+15)/16, (m+15)/16), dim3(16, 16)>>>(d_i, d_w, d_b, d_o, m, n, k, sharpen);
    }
}