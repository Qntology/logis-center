#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Optimized Warp Reduction for sum
__inline__ __device__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

extern "C" __global__ void paged_flash_decoding_bf16_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* const* __restrict__ k_blocks,
    const __nv_bfloat16* const* __restrict__ v_blocks,
    const int* __restrict__ block_lens, 
    __nv_bfloat16* __restrict__ out,
    const int num_blocks,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    const int max_block_size
) {
    int head_idx = blockIdx.x;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int tid = threadIdx.x; 

    // Shared Memory allocation
    extern __shared__ float s_data[];
    __nv_bfloat16* s_query = (__nv_bfloat16*)s_data;
    float* s_reduce = (float*)&s_query[head_dim];

    // 1. Copy Query to Shared Memory
    if (tid < head_dim) {
        s_query[tid] = query[head_idx * head_dim + tid];
    }
    __syncthreads();

    // Online Softmax variables initialization
    float m_i = -1.0e38f;
    float l_i = 0.0f;
    
    // Register for accumulating output V
    float out_acc = 0.0f;

    // 2. Iterate over Paged KV Blocks
    for (int b = 0; b < num_blocks; ++b) {
        const __nv_bfloat16* k_block = k_blocks[b];
        const __nv_bfloat16* v_block = v_blocks[b];

        const int actual_len = block_lens[b]; 

        for (int t = 0; t < actual_len; ++t) {
            float qk = 0.0f;
            int token_offset = (kv_head_idx * max_block_size * head_dim) + (t * head_dim);
            
            if (tid < head_dim) {
                float q_val = __bfloat162float(s_query[tid]);
                float k_val = __bfloat162float(k_block[token_offset + tid]);
                qk = q_val * k_val;
            }

            // Block-level Reduction (128 threads)
            qk = warpReduceSum(qk); 
            if (tid % 32 == 0) {
                s_reduce[tid / 32] = qk; 
            }
            __syncthreads();
            
            float qk_sum = (tid < (blockDim.x / 32)) ? s_reduce[tid] : 0.0f;
            if (tid < 32) {
                qk_sum = warpReduceSum(qk_sum);
            }
            qk_sum = __shfl_sync(0xffffffff, qk_sum, 0) * scale;

            // Online Softmax
            float m_ij = (m_i > qk_sum) ? m_i : qk_sum;
            float p = expf(qk_sum - m_ij);
            float exp_diff = expf(m_i - m_ij);

            l_i = l_i * exp_diff + p;
            m_i = m_ij;

            // Accumulate V
            if (tid < head_dim) {
                float v_val = __bfloat162float(v_block[token_offset + tid]);
                out_acc = out_acc * exp_diff + p * v_val;
            }
        }
    }

    // 3. Final normalization and write to VRAM
    if (tid < head_dim) {
        float final_val = out_acc / l_i;
        out[head_idx * head_dim + tid] = __float2bfloat16(final_val);
    }
}

extern "C" void launch_paged_flash_decoding_wrapper(
    const void* query, const void* k_blocks, const void* v_blocks, 
    const int* block_lens, void* out, 
    int num_blocks, int num_heads, int num_kv_heads, int head_dim, float scale, int block_size
) {
    dim3 grid(num_heads);
    dim3 block(head_dim); 
    
    int shared_mem = (head_dim * sizeof(__nv_bfloat16)) + (32 * sizeof(float)); 
    
    paged_flash_decoding_bf16_kernel<<<grid, block, shared_mem>>>(
        (const __nv_bfloat16*)query, (const __nv_bfloat16* const*)k_blocks, 
        (const __nv_bfloat16* const*)v_blocks, block_lens, (__nv_bfloat16*)out,
        num_blocks, num_heads, num_kv_heads, head_dim, scale, block_size
    );
}
