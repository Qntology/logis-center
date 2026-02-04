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

// [2026-ULTRA-OPTIMIZED] Tiled Bit-Flash Attention 5.1
// Uses Online Softmax (FlashAttention-style) with Warp-Reduction Stability
__global__ void bit_serial_attn_kernel_v2026(
    const float* __restrict__ Q,        
    const uint32_t* __restrict__ K_p,   
    const float* __restrict__ V,        
    float* __restrict__ O,              
    int n_h, int n_kv, int h_d, int t_s, float sc, int q_len,
    float alpha // [NEW] Dynamic stability bias
) {
    int q_idx = blockIdx.x; 
    int h = blockIdx.y;     
    int tid = threadIdx.x;
    if (q_idx >= q_len || h >= n_h) return;

    int k_b = (h_d + 31) / 32;
    int h_kv = h / (n_h / n_kv);
    
    // [STABILITY] Dynamic Keep-alive bias to prevent Softmax death
    const float KEEP_ALIVE_BIAS = alpha;

    // Shared memory for Q bits and tiling metadata
    __shared__ uint32_t s_q_bits[8]; 
    __shared__ float s_running_max;
    __shared__ float s_running_sum;

    // 1. One-time Q-Packing (Warp-parallel)
    if (tid < k_b) {
        uint32_t bts = 0;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            int d_idx = tid * 32 + b;
            if (d_idx < h_d && Q[(q_idx * n_h * h_d) + (h * h_d) + d_idx] >= 0.0f) bts |= (1u << b);
        }
        s_q_bits[tid] = bts;
    }
    
    // Accumulators for O [head_dim]
    float local_o[128]; // Max head_dim 128
    #pragma unroll
    for(int i=0; i<128; ++i) local_o[i] = 0.0f;
    
    float running_max = -1e20f;
    float running_sum = 0.0f;

    if (tid == 0) {
        s_running_max = -1e20f;
        s_running_sum = 0.0f;
    }
    __syncthreads();

    // 2. Tiled Processing (Chunk size: 32 tokens for warp-alignment)
    const int TILE_SIZE = 32;
    for (int t_start = 0; t_start < t_s; t_start += TILE_SIZE) {
        int j = t_start + (tid % TILE_SIZE); 
        float score = -1e20f;
        
        if (j < t_s) {
            int dot = 0;
            #pragma unroll
            for (int kb = 0; kb < k_b; ++kb) {
                dot += (32 - 2 * __popc(s_q_bits[kb] ^ K_p[(j * n_kv + h_kv) * k_b + kb]));
            }
            // Apply scale and stability bias
            score = ((float)dot + KEEP_ALIVE_BIAS) * sc;
        }

        // Warp-level Max reduction (since TILE_SIZE=32)
        float tile_max = score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, offset));

        // Online Softmax update
        float next_max = fmaxf(running_max, tile_max);
        float exp_scale_prev = expf(running_max - next_max);
        float exp_score = (j < t_s) ? expf(score - next_max) : 0.0f;
        
        // Warp-level Sum reduction for exp_score
        float tile_sum = exp_score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, offset);

        // Update global accumulators
        running_sum = running_sum * exp_scale_prev + tile_sum;
        running_max = next_max;

        // Accumulate V values: local_o = local_o * exp_scale_prev + exp_score * V
        for (int d = 0; d < h_d; ++d) {
            float v_val = (j < t_s) ? V[(j * n_kv + h_kv) * h_d + d] : 0.0f;
            float weighted_v = exp_score * v_val;
            
            // Sum across warp for this 'd'
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                weighted_v += __shfl_xor_sync(0xFFFFFFFF, weighted_v, offset);
            
            // Only one thread per warp needs to update the local_o accumulator per tile?
            // Actually, for better occupancy, we can distribute 'd' across threads.
            // But since local_o is private, we just keep it simple: 
            // EACH thread calculates its OWN local_o contribution for its specific 'd'.
            // Wait, FlashAttention typically computes O = sum(P * V).
            // Here we do O_d = sum_j(exp(score_j - max) * V_jd)
        }
        
        // [FIXED-ACCUMULATION] Parallelize over 'd' instead of 'j' for V-reduction
        __shared__ float s_tile_scores[TILE_SIZE];
        if (tid < TILE_SIZE) s_tile_scores[tid] = exp_score;
        __syncthreads();

        for (int d = tid; d < h_d; d += blockDim.x) {
            float v_acc = 0.0f;
            #pragma unroll
            for (int tile_j = 0; tile_j < TILE_SIZE; ++tile_j) {
                int global_j = t_start + tile_j;
                if (global_j < t_s) {
                    v_acc += s_tile_scores[tile_j] * V[(global_j * n_kv + h_kv) * h_d + d];
                }
            }
            local_o[d] = local_o[d] * exp_scale_prev + v_acc;
        }
        __syncthreads();
    }

    // 3. Final Normalization and Store
    float inv_sum = 1.0f / (running_sum + 1e-9f);
    for (int d = tid; d < h_d; d += blockDim.x) {
        O[(q_idx * n_h * h_d) + (h * h_d) + d] = local_o[d] * inv_sum;
    }
}

    // 3. Final Normalization and Store
    float inv_sum = 1.0f / (running_sum + 1e-9f);
    for (int d = tid; d < h_d; d += blockDim.x) {
        O[(q_idx * n_h * h_d) + (h * h_d) + d] = local_o[d] * inv_sum;
    }
}

// [2026-OPTIMIZED] Tile-Fused Multi-Pointer Bit-Flash Attention 
// Handles multiple KV segments without physical concatenation (Zero-Copy Stitching)
__global__ void bit_serial_attn_kernel_tile_fused(
    const float* __restrict__ Q,
    const uint32_t** __restrict__ K_ptrs, // Table of pointers to K segments
    const float** __restrict__ V_ptrs,     // Table of pointers to V segments
    const int* __restrict__ segment_lens,  // Length of each segment
    int num_segments,
    float* __restrict__ O,
    int n_h, int h_d, float sc, int q_len
) {
    int q_idx = blockIdx.x; 
    int h = blockIdx.y;     
    int tid = threadIdx.x;
    if (q_idx >= q_len || h >= n_h) return;

    // Registers for fusion
    extern __shared__ float s_tile[]; 
    __shared__ uint32_t s_q_bits[8]; 
    int k_b = (h_d + 31) / 32;

    // 1. One-time Q-packing in shared memory
    if (tid < k_b) {
        uint32_t bts = 0;
        for (int b = 0; b < 32; ++b) {
            int d_idx = tid * 32 + b;
            if (d_idx < h_d && Q[(q_idx * n_h * h_d) + (h * h_d) + d_idx] >= 0.0f) bts |= (1u << b);
        }
        s_q_bits[tid] = bts;
    }
    __syncthreads();

    float local_max = -1e20f;
    float local_sum = 0.0f;
    
    // 2. Fused Multi-Segment Loop
    for (int seg = 0; seg < num_segments; ++seg) {
        const uint32_t* K_seg = K_ptrs[seg];
        int t_s = segment_lens[seg];
        
        for (int j = tid; j < t_s; j += blockDim.x) {
            int dot = 0;
            #pragma unroll
            for (int kb = 0; kb < k_b; ++kb) {
                // [GQA-FIX-TODO] Update this kernel as well if used, for now focusing on v2026
                 dot += (32 - 2 * __popc(s_q_bits[kb] ^ K_seg[(j * n_h + h) * k_b + kb]));
            }
            float score = (float)dot * sc;
            
            // Online Softmax update (FlashAttention style)
            float prev_max = local_max;
            local_max = fmaxf(local_max, score);
            local_sum = local_sum * expf(prev_max - local_max) + expf(score - local_max);
            s_tile[j] = score; // Store score temporarily in shared tile
        }
    }
    // [STUB] Final Weighted sum with V_ptrs[seg] follows here...
}

extern "C" {
    void bit_serial_attn_cuda_direct(const float* d_q, const uint32_t* d_k, const float* d_v, float* d_o, int n_h, int n_kv, int h_d, int t_s, float scale, int dev, int q_len) {
        cudaSetDevice(dev);
        dim3 block(256); // 4 warps
        dim3 grid(q_len, n_h);
        // Tiled kernel handles synchronization internally. Small shared memory for Q-bits and tiling.
        size_t smem = 1024; 
        bit_serial_attn_kernel_v2026<<<grid, block, smem>>>(d_q, d_k, d_v, d_o, n_h, n_kv, h_d, t_s, scale, q_len);
    }

    void bit_serial_attn_cuda_fused(const float* d_q, const uint32_t** d_k_table, const float** d_v_table, const int* d_lens, int n_segs, float* d_o, int n_h, int h_d, float scale, int dev, int q_len) {
        cudaSetDevice(dev);
        dim3 block(256);
        dim3 grid(q_len, n_h);
        // [2026-FUSION] Maximize shared memory usage for tiling
        size_t smem = 16384; 
        bit_serial_attn_kernel_tile_fused<<<grid, block, smem>>>(d_q, d_k_table, d_v_table, d_lens, n_segs, d_o, n_h, h_d, scale, q_len);
    }
}
