#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp 내에서 합계를 구하는 최적화 함수 (Warp Reduction)
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
    const int* __restrict__ block_lens, // 추가: 각 블록의 실제 토큰 수
    __nv_bfloat16* __restrict__ out,
    const int num_blocks,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    const int max_block_size
) {
    // blockIdx.x는 쿼리의 Attention Head 하나를 전담합니다.
    int head_idx = blockIdx.x;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads); // GQA 인덱스 매핑
    int tid = threadIdx.x; // 스레드 수는 head_dim (예: 128)과 동일하게 세팅

    // Shared Memory 할당 (Query 캐싱 및 워프 간 통신용)
    extern __shared__ float s_data[];
    __nv_bfloat16* s_query = (__nv_bfloat16*)s_data;
    float* s_reduce = (float*)&s_query[head_dim];

    // 1. Query를 빠르고 반복적인 접근을 위해 Shared Memory에 복사
    if (tid < head_dim) {
        s_query[tid] = query[head_idx * head_dim + tid];
    }
    __syncthreads();

    // Online Softmax 변수 초기화
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    
    // 최종 출력 V를 누적할 로컬 레지스터
    float out_acc = 0.0f;

    // 2. Paged KV Blocks 순회
    for (int b = 0; b < num_blocks; ++b) {
        const __nv_bfloat16* k_block = k_blocks[b];
        const __nv_bfloat16* v_block = v_blocks[b];

        const int actual_len = block_lens[b]; // 이번 블록의 실제 길이

        // 👇 max_block_size 대신 actual_len 사용으로 메모리 초과 방지
        for (int t = 0; t < actual_len; ++t) {
            float qk = 0.0f;
            // offset 계산 시 실제 메모리 구조 [heads, len, dim] 고려
            int token_offset = (kv_head_idx * actual_len * head_dim) + (t * head_dim);
            
            if (tid < head_dim) {
                float q_val = __bfloat162float(s_query[tid]);
                float k_val = __bfloat162float(k_block[token_offset + tid]);
                qk = q_val * k_val;
            }

            // --- Block-level Reduction (128개 스레드의 합을 구함) ---
            qk = warpReduceSum(qk); // 워프(32) 내 합산
            if (tid % 32 == 0) {
                s_reduce[tid / 32] = qk; // 워프 리더가 Shared Memory에 기록
            }
            __syncthreads();
            
            // 첫 번째 워프가 나머지 워프들의 결과를 합산
            float qk_sum = (tid < (blockDim.x / 32)) ? s_reduce[tid] : 0.0f;
            if (tid < 32) {
                qk_sum = warpReduceSum(qk_sum);
            }
            // 0번 스레드가 가진 최종 합계를 모든 스레드에 브로드캐스트
            qk_sum = __shfl_sync(0xffffffff, qk_sum, 0) * scale;
            // --------------------------------------------------------

            // [B] Online Softmax 계산
            float m_ij = max(m_i, qk_sum);
            float p = expf(qk_sum - m_ij);
            float exp_diff = expf(m_i - m_ij);

            l_i = l_i * exp_diff + p;
            m_i = m_ij;

            // [C] V 누산 (각 스레드가 자신의 차원(d)에 해당하는 V 값만 계산)
            if (tid < head_dim) {
                float v_val = __bfloat162float(v_block[token_offset + tid]);
                out_acc = out_acc * exp_diff + p * v_val;
            }
        }
    }

    // 3. 최종 정규화 및 BF16 변환 후 VRAM에 쓰기
    if (tid < head_dim) {
        float final_val = out_acc / l_i;
        out[head_idx * head_dim + tid] = __float2bfloat16(final_val);
    }
}

// --- paged_flash_decoding.cu 파일의 맨 끝에 추가 ---

extern "C" void launch_paged_flash_decoding_wrapper(
    const void* query, const void* k_blocks, const void* v_blocks, 
    const int* block_lens, void* out, // block_lens 추가
    int num_blocks, int num_heads, int num_kv_heads, int head_dim, float scale, int block_size
) {
    dim3 grid(num_heads);
    dim3 block(head_dim); // 스레드 수를 head_dim(예: 128)으로 맞춤
    
    // Shared memory: query(BF16) + 리덕션용 float 공간
    int shared_mem = (head_dim * sizeof(__nv_bfloat16)) + (32 * sizeof(float)); 
    
    paged_flash_decoding_bf16_kernel<<<grid, block, shared_mem>>>(
        (const __nv_bfloat16*)query, (const __nv_bfloat16* const*)k_blocks, 
        (const __nv_bfloat16* const*)v_blocks, block_lens, (__nv_bfloat16*)out,
        num_blocks, num_heads, num_kv_heads, head_dim, scale, block_size
    );
}