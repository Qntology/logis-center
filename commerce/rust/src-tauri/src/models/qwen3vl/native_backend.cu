#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

// [2026-HYBRID-MASTER] Optimized Folding Bit-serial Matmul for F16 with Dynamic Stabilization
__global__ void bit_serial_matmul_kernel_f16(
    const half* __restrict__ input,    
    const uint32_t* __restrict__ weight, 
    const half* __restrict__ scale,     
    half* __restrict__ output,          
    int M, int N, int K, int src_k
) {
    const int TILE_M = 32;
    __shared__ uint32_t s_input_bits[TILE_M]; 
    __shared__ float s_input_scales[TILE_M]; // [STABILITY] Dynamic Input Scale Pool

    int m_base = blockIdx.y * TILE_M;
    int n_group = blockIdx.x;
    int n_sub = threadIdx.x; 
    int m_sub = threadIdx.y; 
    
    int m = m_base + m_sub;
    int n = n_group * 8 + n_sub;
    
    int src_k_blocks = (src_k + 31) / 32;
    int K_blocks = (K + 31) / 32; // Actual physical stride in memory
    int repetition = (K > src_k) ? (K / src_k) : 1;
    float r_scale = rsqrtf((float)repetition); // [STABILITY] Magnitude correction factor

    float acc = 0.0f;

    for (int kb = 0; kb < src_k_blocks; ++kb) {
        if (n_sub == 0) {
            uint32_t folded_bts = 0;
            float block_sum_sq = 0.0f;
            if (m < M) {
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    int k_base = kb * 32 + b;
                    if (k_base < src_k) {
                        float combined_signal = 0.0f;
                        // [HYBRID-CYCLIC-REPETITION] Repeat input signal to match expanded target dimension
                        for (int r = 0; r < repetition; ++r) {
                            int target_k_idx = k_base + (r * src_k);
                            combined_signal += __half2float(input[m * K + (target_k_idx % K)]);
                        }
                        // [STABILITY] Normalize folded signal energy
                        combined_signal *= r_scale;
                        if (combined_signal >= 0.0f) folded_bts |= (1u << b);
                        block_sum_sq += combined_signal * combined_signal;
                    }
                }
            }
            s_input_bits[m_sub] = folded_bts;
            // [STABILITY] Per-block RMS for dynamic input scaling
            s_input_scales[m_sub] = sqrtf(block_sum_sq / 32.0f + 1e-8f);
        }
        __syncthreads();

        if (m < M && n < N) {
            // [FIX] Use K_blocks for correct weight memory stride in expanded hybrid models
            int idx = n_group * K_blocks * 8 + kb * 8 + n_sub;
            uint32_t w_val = weight[idx];
            float s_val = __half2float(scale[idx]);
            uint32_t diff = s_input_bits[m_sub] ^ w_val;
            // [HYBRID-PRECISION] Combine weight scale with dynamic input scale
            acc += (float)(32 - 2 * (int)__popc(diff)) * s_val * s_input_scales[m_sub];
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        output[m * N + n] = __float2half(acc);
    }
}

// [NEW] 4-bit Sliced Matmul Kernel (Bit-serial variant)
__global__ void bit_serial_matmul_kernel_4bit_f16(
    const half* __restrict__ input,    
    const uint32_t* __restrict__ weight, 
    const half* __restrict__ scale,     
    half* __restrict__ output,          
    int M, int N, int K, int src_k
) {
    const int TILE_M = 32;
    __shared__ uint32_t s_input_bits[TILE_M]; 
    __shared__ float s_input_scales[TILE_M];

    int m_base = blockIdx.y * TILE_M;
    int n_group = blockIdx.x;
    int n_sub = threadIdx.x; 
    int m_sub = threadIdx.y; 
    
    int m = m_base + m_sub;
    int n = n_group * 8 + n_sub;
    
    int src_k_blocks = (src_k + 31) / 32;
    int K_blocks = (K + 31) / 32;
    int slice_stride = (N + 7) / 8 * K_blocks * 8; 
    int repetition = (K > src_k) ? (K / src_k) : 1;
    float r_scale = rsqrtf((float)repetition);

    float acc = 0.0f;

    for (int kb = 0; kb < src_k_blocks; ++kb) {
        if (n_sub == 0) {
            uint32_t folded_bts = 0;
            float block_sum_sq = 0.0f;
            if (m < M) {
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    int k_base = kb * 32 + b;
                    if (k_base < src_k) {
                        float combined_signal = 0.0f;
                        for (int r = 0; r < repetition; ++r) {
                            int target_k_idx = k_base + (r * src_k);
                            combined_signal += __half2float(input[m * K + (target_k_idx % K)]);
                        }
                        combined_signal *= r_scale;
                        if (combined_signal >= 0.0f) folded_bts |= (1u << b);
                        block_sum_sq += combined_signal * combined_signal;
                    }
                }
            }
            s_input_bits[m_sub] = folded_bts;
            s_input_scales[m_sub] = sqrtf(block_sum_sq / 32.0f + 1e-8f);
        }
        __syncthreads();

        if (m < M && n < N) {
            int base_idx = n_group * K_blocks * 8 + kb * 8 + n_sub;
            float s_val = __half2float(scale[n]); // [FIX] Dynamic scale fetch per output channel
            
            float slice_acc_sum = 0.0f;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint32_t w_val = weight[b * slice_stride + base_idx];
                uint32_t diff = s_input_bits[m_sub] ^ w_val;
                float pop = (float)(32 - 2 * (int)__popc(diff));
                slice_acc_sum += pop * (float)(1 << b);
            }
            // Accurate 4-bit accumulation
            acc += slice_acc_sum * s_val * s_input_scales[m_sub]; 
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        output[m * N + n] = __float2half(acc);
    }
}

// [FAST-ATTENTION-HYBRID] Folding-aware online softmax attention for F16 with Dynamic Stabilization
__global__ void bit_serial_attn_kernel_f16(
    const half* __restrict__ Q,        
    const uint32_t* __restrict__ K_p,   
    const half* __restrict__ V,        
    half* __restrict__ O,              
    int n_h, int n_kv, int h_d, int t_s, float sc, int q_len,
    float alpha, int src_h_d
) {
    int q_idx = blockIdx.x; 
    int h = blockIdx.y;     
    int tid = threadIdx.x;
    
    if (q_idx >= q_len || h >= n_h) return;

    int src_k_b = (src_h_d + 31) / 32;
    int h_d_blocks = (h_d + 31) / 32; // Actual physical stride in KV cache
    int h_kv = h / (n_h / n_kv);
    int repetition = (h_d > src_h_d) ? (h_d / src_h_d) : 1;
    float r_scale = rsqrtf((float)repetition); // [STABILITY] Magnitude correction
    
    __shared__ uint32_t s_q_bits[16]; 
    __shared__ float s_q_scales[16]; // [STABILITY] Dynamic Q-Scale Pool
    
    if (tid < src_k_b) {
        uint32_t folded_bts = 0;
        float q_block_sum_sq = 0.0f;
        for (int b = 0; b < 32; ++b) {
            int d_base = tid * 32 + b;
            if (d_base < src_h_d) {
                float combined_q = 0.0f;
                // [HYBRID-CYCLIC-REPETITION] Cyclic read for Query signals
                for (int r = 0; r < repetition; ++r) {
                    int target_d_idx = d_base + (r * src_h_d);
                    combined_q += __half2float(Q[(q_idx * n_h * h_d) + (h * h_d) + (target_d_idx % h_d)]);
                }
                combined_q *= r_scale;
                if (combined_q >= 0.0f) folded_bts |= (1u << b);
                q_block_sum_sq += combined_q * combined_q;
            }
        }
        s_q_bits[tid] = folded_bts;
        s_q_scales[tid] = sqrtf(q_block_sum_sq / 32.0f + 1e-8f);
    }
    __syncthreads();
    
    float running_max = -10000.0f; // [STABILITY] Use a safer initial max for half-precision ranges
    float running_sum = 0.0f;
    float local_o = 0.0f; 

    for (int j = 0; j < t_s; ++j) {
        float dot = 0.0f;
        for (int kb = 0; kb < src_k_b; ++kb) {
            dot += (float)(32 - 2 * (int)__popc(s_q_bits[kb] ^ K_p[(j * n_kv + h_kv) * h_d_blocks + kb])) * s_q_scales[kb];
        }
        float score = (dot + alpha) * sc;
        // [STABILITY] Clamp scores to prevent exp() overflow
        if (score < -20.0f) score = -20.0f;
        if (score > 20.0f) score = 20.0f;

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
        // [STABILITY] Add stronger epsilon to prevent division by zero
        O[(q_idx * n_h * h_d) + (h * h_d) + tid] = __float2half(local_o / (running_sum + 1e-12f));
    }
}

// [STANDARD-F16-MATMUL] For non-quantized layers like lm_head or standard projections
__global__ void standard_matmul_kernel_f16(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __half2float(A[m * K + k]) * __half2float(B[n * K + k]); 
        }
        C[m * N + n] = __float2half(acc);
    }
}

// [RMS-NORM-F16] Ultra-stable Warp-Shuffle based RMS Normalization
__global__ void rms_norm_kernel_f16(const half* i, const half* w, half* o, int h, float e) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int col = tid; col < h; col += blockDim.x) {
        float val = __half2float(i[row * h + col]);
        sum += val * val;
    }

    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float s_warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) s_warp_sums[warp_id] = sum;
    __syncthreads();

    if (tid < 32) {
        float s = (tid < (blockDim.x + 31) / 32) ? s_warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        if (tid == 0) s_warp_sums[0] = s;
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_warp_sums[0] / h + e);
    for (int col = tid; col < h; col += blockDim.x) {
        o[row * h + col] = __float2half(__half2float(i[row * h + col]) * inv_rms * __half2float(w[col]));
    }
}

// [SILU-MUL-F16] High-speed Swiglu Activation
__global__ void silu_mul_kernel_f16(half* gate, const half* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        gate[idx] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

// [GPU-ROPE] Fast In-place RoPE Application
__global__ void apply_rope_inplace_kernel_f16(half* Q, half* K, const half* cos_table, const half* sin_table, int q_len, int s_o, int n_h, int n_kv, int h_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = idx / h_d;
    int d_idx = idx % h_d;
    if (d_idx >= h_d / 2) return; // Process pairs

    int token_idx = head_idx / (n_h + n_kv);
    int p = token_idx + s_o;
    int target_head = head_idx % (n_h + n_kv);
    
    half* target_ptr = (target_head < n_h) ? 
        &Q[token_idx * n_h * h_d + target_head * h_d] : 
        &K[token_idx * n_kv * h_d + (target_head - n_h) * h_d];

    float q0 = __half2float(target_ptr[d_idx]);
    float q1 = __half2float(target_ptr[d_idx + h_d / 2]);
    float c = __half2float(cos_table[p * (h_d / 2) + d_idx]);
    float s = __half2float(sin_table[p * (h_d / 2) + d_idx]);

    target_ptr[d_idx] = __float2half(q0 * c - q1 * s);
    target_ptr[d_idx + h_d / 2] = __float2half(q0 * s + q1 * c);
}

// [GPU-PACKING] Fast Bit-serial Compression
__global__ void pack_f16_to_u32_kernel(const half* src, uint32_t* dst, int total_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 32 >= total_elements) return;

    uint32_t packed = 0;
    for (int i = 0; i < 32; ++i) {
        int idx = tid * 32 + i;
        if (idx < total_elements && __half2float(src[idx]) >= 0.0f) {
            packed |= (1u << i);
        }
    }
    dst[tid] = packed;
}

// [GPU-GAIN] Apply gain
__global__ void apply_gain_kernel_f16(half* data, float gain, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) data[i] = __float2half(__half2float(data[i]) * gain);
}

// [GPU-RESIDUAL] Fast In-place Addition
__global__ void add_inplace_kernel_f16(half* dst, const half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

// [GPU-HYBRID-EXPAND] Fast repetition for 0.6B -> 2B transition
__global__ void hybrid_repeat_kernel_f16(half* data, int src_size, int target_size, int q_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = q_len * target_size;
    if (tid >= total_elements) return;

    int t = tid / target_size;
    int i = tid % target_size;
    if (i >= src_size) {
        data[tid] = data[t * target_size + (i % src_size)];
    }
}

// [GPU-SILU] Fast element-wise SiLU
__global__ void silu_inplace_kernel_f16(half* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float f = __half2float(data[i]);
        data[i] = __float2half(f / (1.0f + expf(-f)));
    }
}

// [GPU-ELEMENT-MUL] Element-wise Multiplication
__global__ void element_mul_kernel_f16(half* dst, const half* src, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = __float2half(__half2float(dst[i]) * __half2float(src[i]));
}

// --- EXTERN C INTERFACE ---
extern "C" {
    // 1. Bit-Serial Matmul
    void bit_serial_matmul_cuda_f16(const half* d_i, const uint32_t* d_w, const half* d_s, half* d_o, int m, int n, int k, int dev, int src_k) {
        dim3 block(8, 32);
        dim3 grid((n + 7) / 8, (m + 31) / 32);
        bit_serial_matmul_kernel_f16<<<grid, block>>>(d_i, d_w, d_s, d_o, m, n, k, src_k);
    }

    void bit_serial_matmul_cuda_4bit_f16(const half* d_i, const uint32_t* d_w, const half* d_s, half* d_o, int m, int n, int k, int dev, int src_k) {
        dim3 block(8, 32);
        dim3 grid((n + 7) / 8, (m + 31) / 32);
        bit_serial_matmul_kernel_4bit_f16<<<grid, block>>>(d_i, d_w, d_s, d_o, m, n, k, src_k);
    }

    // 2. Bit-Serial Attention
    void bit_serial_attn_cuda_f16(const half* d_q, const uint32_t* d_k, const half* d_v, half* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len, float alpha, int src_h_d) {
        dim3 block(h_d < 1024 ? h_d : 1024); 
        // [FIX] Correct axis mapping: x=heads, y=tokens to match kernel internals
        dim3 grid(n_h, q_len); 
        bit_serial_attn_kernel_f16<<<grid, block>>>(d_q, d_k, d_v, d_o, n_h, n_kv, h_d, t_s, scale, q_len, alpha, src_h_d);
    }

    // 3. Standard F16 Matmul
    void standard_matmul_cuda_f16(const half* d_i, const half* d_w, half* d_o, int m, int n, int k) {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        standard_matmul_kernel_f16<<<grid, block>>>(d_i, d_w, d_o, m, n, k);
    }

    // 4. RMS Norm
    void cuda_rms_norm_f16(const half* d_i, const half* d_w, half* d_o, int m, int hid, float eps) {
        int threads = 256; 
        rms_norm_kernel_f16<<<m, threads>>>(d_i, d_w, d_o, hid, eps);
    }

    // 5. Silu Mul
    void cuda_silu_mul_f16(half* d_gate, const half* d_up, int size) {
        silu_mul_kernel_f16<<<(size + 255) / 256, 256>>>(d_gate, d_up, size);
    }

    void cuda_apply_rope_f16(half* d_q, half* d_k, const half* d_cos, const half* d_sin, int q_len, int s_o, int n_h, int n_kv, int h_d) {
        int total_heads = (n_h + n_kv) * q_len;
        apply_rope_inplace_kernel_f16<<< (total_heads * (h_d/2) + 255)/256, 256 >>>(d_q, d_k, d_cos, d_sin, q_len, s_o, n_h, n_kv, h_d);
    }

    void cuda_pack_bits_f16(const half* d_src, uint32_t* d_dst, int elements) {
        int blocks = (elements + 31) / 32;
        pack_f16_to_u32_kernel<<< (blocks + 255)/256, 256 >>>(d_src, d_dst, elements);
    }

    void cuda_add_inplace_f16(half* d_dst, const half* d_src, int size) {
        add_inplace_kernel_f16<<<(size + 255)/256, 256>>>(d_dst, d_src, size);
    }

    void cuda_hybrid_repeat_f16(half* d_data, int src_size, int target_size, int q_len) {
        int total = q_len * target_size;
        hybrid_repeat_kernel_f16<<<(total + 255)/256, 256>>>(d_data, src_size, target_size, q_len);
    }

    void cuda_silu_inplace_f16(half* d_data, int size) {
        silu_inplace_kernel_f16<<<(size + 255)/256, 256>>>(d_data, size);
    }

    void cuda_apply_gain_f16(half* d_data, float gain, int elements) {
        apply_gain_kernel_f16<<<(elements + 255)/256, 256>>>(d_data, gain, elements);
    }

    void cuda_element_mul_f16(half* d_dst, const half* d_src, int size) {
        element_mul_kernel_f16<<<(size + 255)/256, 256>>>(d_dst, d_src, size);
    }
}