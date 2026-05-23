#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>

// 최신 LLM(Qwen 등)에서 주로 사용하는 최대 Head Dimension 크기
#define MAX_HEAD_DIM 128

// Flash Attention v1의 핵심 원리인 Online Softmax를 레지스터 레벨에서 구현한 커널
// [최적화] half2 벡터화를 통해 VRAM 대역폭 효율을 2배로 극대화
__global__ void online_softmax_attention_f16_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int seq_len,
    int head_dim,
    float scale
) {
    // 2D Grid를 사용하여 seq_len이 1024를 초과하더라도 안정적으로 스레드 분배
    int batch_head_idx = blockIdx.x; 
    int q_idx = blockIdx.y * blockDim.x + threadIdx.x; 

    if (q_idx >= seq_len) return;

    int base_offset = batch_head_idx * seq_len * head_dim;
    
    // 메모리 1회 접근량을 16bit -> 32bit(half2)로 2배 증가시키기 위한 포인터 캐스팅
    const half2* q_row = reinterpret_cast<const half2*>(q + base_offset + q_idx * head_dim);
    half2* out_row = reinterpret_cast<half2*>(out + base_offset + q_idx * head_dim);

    // Flash Attention: Shared Memory 대신 레지스터에 누적값을 보관하여 메모리 한계 극복
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float out_val[MAX_HEAD_DIM] = {0.0f};

    // Query 토큰을 레지스터로 미리 로드 (루프 횟수 50% 절감)
    int half_head_dim = head_dim / 2;
    float2 q_reg[MAX_HEAD_DIM / 2];
    for(int d = 0; d < half_head_dim; ++d) {
        q_reg[d] = __half22float2(q_row[d]);
    }

    // Key, Value를 순회하며 Online Softmax 계산
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        const half2* k_row = reinterpret_cast<const half2*>(k + base_offset + k_idx * head_dim);
        const half2* v_row = reinterpret_cast<const half2*>(v + base_offset + k_idx * head_dim);
        
        // 1. Score 계산: Q_i * K_j^T * scale (2차원 벡터 내적 동시 처리)
        float s_ij = 0.0f;
        for (int d = 0; d < half_head_dim; ++d) {
            float2 k_val = __half22float2(k_row[d]);
            s_ij += q_reg[d].x * k_val.x + q_reg[d].y * k_val.y;
        }
        s_ij *= scale;

        // 2. Online Softmax 수학 공식 적용 (OOM 원천 차단)
        float m_new = max(m_i, s_ij);
        float exp_old = expf(m_i - m_new);
        float exp_new = expf(s_ij - m_new);

        l_i = l_i * exp_old + exp_new;

        // 3. V 값 누적 (2개의 출력을 동시 융합 연산)
        for (int d = 0; d < half_head_dim; ++d) {
            float2 v_val = __half22float2(v_row[d]);
            out_val[d * 2] = out_val[d * 2] * exp_old + exp_new * v_val.x;
            out_val[d * 2 + 1] = out_val[d * 2 + 1] * exp_old + exp_new * v_val.y;
        }
        
        m_i = m_new;
    }

    // 4. 최종 정규화 및 Global Memory 출력 (half2로 한 번에 쓰기)
    for (int d = 0; d < half_head_dim; ++d) {
        float2 res;
        res.x = out_val[d * 2] / l_i;
        res.y = out_val[d * 2 + 1] / l_i;
        out_row[d] = __float22half2_rn(res);
    }
}

extern "C" {
    void flash_attn(
        const void* q_ptr,
        const void* k_ptr,
        const void* v_ptr,
        void* out_ptr,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        float softmax_scale
    ) {
        // block_dim이 1024를 넘어가면 터지므로 256으로 고정하고 Grid_Y로 분할합니다.
        dim3 block_dim(256);
        dim3 grid_dim(batch_size * num_heads, (seq_len + block_dim.x - 1) / block_dim.x);
        
        // Shared Memory(0) 없이 실행 가능
        online_softmax_attention_f16_kernel<<<grid_dim, block_dim, 0>>>(
            reinterpret_cast<const half*>(q_ptr),
            reinterpret_cast<const half*>(k_ptr),
            reinterpret_cast<const half*>(v_ptr),
            reinterpret_cast<half*>(out_ptr),
            seq_len,
            head_dim,
            softmax_scale
        );
    }
}



__global__ void fused_rope_f16_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    int seq_len,
    int q_heads,
    int k_heads,
    int head_dim,
    int batch_size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int half_dim = head_dim / 2;
    int total_elements = batch_size * seq_len * q_heads * half_dim;

    if (idx >= total_elements) return;

    if (idx + 1 < total_elements && (idx % half_dim) + 1 < half_dim) {
        int d_idx = idx % half_dim;
        int head_idx = (idx / half_dim) % q_heads;
        int seq_idx = (idx / (half_dim * q_heads)) % seq_len;
        int b_idx = idx / (half_dim * q_heads * seq_len);

        int q_offset1 = b_idx * (seq_len * q_heads * head_dim) + seq_idx * (q_heads * head_dim) + head_idx * head_dim + d_idx;
        int q_offset2 = q_offset1 + half_dim;
        int cos_sin_offset = seq_idx * head_dim + d_idx;

        float2 c2 = __half22float2(*reinterpret_cast<const half2*>(&cos[cos_sin_offset]));
        float2 s2 = __half22float2(*reinterpret_cast<const half2*>(&sin[cos_sin_offset]));
        float2 q1_f2 = __half22float2(*reinterpret_cast<const half2*>(&q[q_offset1]));
        float2 q2_f2 = __half22float2(*reinterpret_cast<const half2*>(&q[q_offset2]));

        float2 res_q1, res_q2;
        res_q1.x = q1_f2.x * c2.x - q2_f2.x * s2.x;
        res_q1.y = q1_f2.y * c2.y - q2_f2.y * s2.y;
        res_q2.x = q2_f2.x * c2.x + q1_f2.x * s2.x;
        res_q2.y = q2_f2.y * c2.y + q1_f2.y * s2.y;

        *reinterpret_cast<half2*>(&q[q_offset1]) = __float22half2_rn(res_q1);
        *reinterpret_cast<half2*>(&q[q_offset2]) = __float22half2_rn(res_q2);

        if (head_idx < k_heads) {
            int k_offset1 = b_idx * (seq_len * k_heads * head_dim) + seq_idx * (k_heads * head_dim) + head_idx * head_dim + d_idx;
            int k_offset2 = k_offset1 + half_dim;
            float2 k1_f2 = __half22float2(*reinterpret_cast<const half2*>(&k[k_offset1]));
            float2 k2_f2 = __half22float2(*reinterpret_cast<const half2*>(&k[k_offset2]));

            float2 res_k1, res_k2;
            res_k1.x = k1_f2.x * c2.x - k2_f2.x * s2.x;
            res_k1.y = k1_f2.y * c2.y - k2_f2.y * s2.y;
            res_k2.x = k2_f2.x * c2.x + k1_f2.x * s2.x;
            res_k2.y = k2_f2.y * c2.y + k1_f2.y * s2.y;

            *reinterpret_cast<half2*>(&k[k_offset1]) = __float22half2_rn(res_k1);
            *reinterpret_cast<half2*>(&k[k_offset2]) = __float22half2_rn(res_k2);
        }
    } else {
        for(int i=0; i<2; ++i) {
            int curr = idx + i;
            if (curr >= total_elements) break;
            int d_idx = curr % half_dim;
            int head_idx = (curr / half_dim) % q_heads;
            int seq_idx = (curr / (half_dim * q_heads)) % seq_len;
            int b_idx = curr / (half_dim * q_heads * seq_len);

            int q_offset1 = b_idx * (seq_len * q_heads * head_dim) + seq_idx * (q_heads * head_dim) + head_idx * head_dim + d_idx;
            int q_offset2 = q_offset1 + half_dim;
            int cos_sin_offset = seq_idx * head_dim + d_idx;

            float c = __half2float(cos[cos_sin_offset]);
            float s = __half2float(sin[cos_sin_offset]);
            float q1 = __half2float(q[q_offset1]);
            float q2 = __half2float(q[q_offset2]);

            q[q_offset1] = __float2half(q1 * c - q2 * s);
            q[q_offset2] = __float2half(q2 * c + q1 * s);

            if (head_idx < k_heads) {
                int k_offset1 = b_idx * (seq_len * k_heads * head_dim) + seq_idx * (k_heads * head_dim) + head_idx * head_dim + d_idx;
                int k_offset2 = k_offset1 + half_dim;
                float k1 = __half2float(k[k_offset1]);
                float k2 = __half2float(k[k_offset2]);
                k[k_offset1] = __float2half(k1 * c - k2 * s);
                k[k_offset2] = __float2half(k2 * c + k1 * s);
            }
        }
    }
}

extern "C" {
    void fused_apply_rotary_pos_emb(
        void* q_ptr,
        void* k_ptr,
        const void* cos_ptr,
        const void* sin_ptr,
        int batch_size,
        int seq_len,
        int q_heads,
        int k_heads,
        int head_dim
    ) {
        int half_dim = head_dim / 2;
        int total_threads = batch_size * seq_len * q_heads * half_dim;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;

        fused_rope_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<half*>(q_ptr),
            reinterpret_cast<half*>(k_ptr),
            reinterpret_cast<const half*>(cos_ptr),
            reinterpret_cast<const half*>(sin_ptr),
            seq_len,
            q_heads,
            k_heads,
            head_dim,
            batch_size
        );
    }
}


__global__ void fused_gated_rmsnorm_f16_kernel(
    const half* __restrict__ xs,
    const half* __restrict__ weight,
    const half* __restrict__ gate,
    half* __restrict__ out,
    float eps,
    int hidden_size,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int row_idx = idx / hidden_size;
    int col_idx = idx % hidden_size;
    int row_start = row_idx * hidden_size;

    float sum_sq = 0.0f;
    int half_hidden = hidden_size / 2;
    const half2* row_ptr = reinterpret_cast<const half2*>(xs + row_start);
    for (int i = 0; i < half_hidden; ++i) {
        float2 val = __half22float2(row_ptr[i]);
        sum_sq += val.x * val.x + val.y * val.y;
    }
    if (hidden_size % 2 != 0) {
        float val = __half2float(xs[row_start + hidden_size - 1]);
        sum_sq += val * val;
    }

    float variance = sum_sq / hidden_size;
    float inv_std = rsqrtf(variance + eps);
    float xs_val = __half2float(xs[idx]);
    float w_val = __half2float(weight[col_idx]);
    float norm_val = xs_val * inv_std * w_val;

    if (gate != nullptr) {
        float gate_val = __half2float(gate[idx]);
        float silu_val = gate_val / (1.0f + expf(-gate_val));
        norm_val *= silu_val;
    }
    out[idx] = __float2half(norm_val);
}

__global__ void fused_silu_mul_f16_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int total_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    if (idx + 1 < total_elements) {
        half2 g2 = *reinterpret_cast<const half2*>(&gate[idx]);
        half2 u2 = *reinterpret_cast<const half2*>(&up[idx]);

        float2 g_f2 = __half22float2(g2);
        float2 u_f2 = __half22float2(u2);

        float sig_g0 = g_f2.x / (1.0f + expf(-g_f2.x));
        float sig_g1 = g_f2.y / (1.0f + expf(-g_f2.y));

        float2 out_f2;
        out_f2.x = sig_g0 * u_f2.x;
        out_f2.y = sig_g1 * u_f2.y;

        *reinterpret_cast<half2*>(&out[idx]) = __float22half2_rn(out_f2);
    } else {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu_g * u);
    }
}

// ==============================================================================
// [Unified FFI Dispatcher Layer] Rust와의 통신을 담당하는 단일 공용 인터페이스
// ==============================================================================

// 공용 Grid 차원 계산 헬퍼 함수
inline void get_launch_config(int total_elements, int& grid_dim, int& block_dim) {
    block_dim = 256;
    grid_dim = (total_elements + block_dim - 1) / block_dim;
}

extern "C" {

    void fused_gated_rmsnorm(
        const void* xs_ptr,
        const void* weight_ptr,
        const void* gate_ptr,
        void* out_ptr,
        float eps,
        int hidden_size,
        int total_elements
    ) {
        int grid_dim, block_dim;
        get_launch_config(total_elements, grid_dim, block_dim);

        fused_gated_rmsnorm_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(xs_ptr),
            reinterpret_cast<const half*>(weight_ptr),
            reinterpret_cast<const half*>(gate_ptr),
            reinterpret_cast<half*>(out_ptr),
            eps,
            hidden_size,
            total_elements
        );
    }

    void fused_silu_mul(
        const void* gate_ptr,
        const void* up_ptr,
        void* out_ptr,
        int total_elements
    ) {
        int total_threads = (total_elements + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;

        fused_silu_mul_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(gate_ptr),
            reinterpret_cast<const half*>(up_ptr),
            reinterpret_cast<half*>(out_ptr),
            total_elements
        );
    }
}



// 1. Fused SSM State Kernel (for Qwen 3.5 Gated Delta Net)
__global__ void fused_ssm_state_f16_kernel(
    const half* __restrict__ b,
    const half* __restrict__ a,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    half* __restrict__ beta_out,
    half* __restrict__ g_out,
    int num_v_heads,
    int total_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    if (idx + 1 < total_elements) {
        half2 b2 = *reinterpret_cast<const half2*>(&b[idx]);
        half2 a2 = *reinterpret_cast<const half2*>(&a[idx]);

        float2 b_f2 = __half22float2(b2);
        float2 a_f2 = __half22float2(a2);

        int h_idx1 = idx % num_v_heads;
        int h_idx2 = (idx + 1) % num_v_heads;

        // Beta = sigmoid(b) 동시 연산
        float2 beta_val;
        beta_val.x = 1.0f / (1.0f + expf(-b_f2.x));
        beta_val.y = 1.0f / (1.0f + expf(-b_f2.y));

        *reinterpret_cast<half2*>(&beta_out[idx]) = __float22half2_rn(beta_val);

        // a_plus_bias = softplus(a + dt_bias) 동시 연산
        float sum_a1 = a_f2.x + dt_bias[h_idx1];
        float sum_a2 = a_f2.y + dt_bias[h_idx2];
        
        float sp_a1 = sum_a1 > 20.0f ? sum_a1 : logf(1.0f + expf(sum_a1));
        float sp_a2 = sum_a2 > 20.0f ? sum_a2 : logf(1.0f + expf(sum_a2));

        float2 g_val;
        g_val.x = a_log[h_idx1] * sp_a1;
        g_val.y = a_log[h_idx2] * sp_a2;

        *reinterpret_cast<half2*>(&g_out[idx]) = __float22half2_rn(g_val);
    } else {
        int h_idx = idx % num_v_heads;
        float b_val = __half2float(b[idx]);
        float beta_val = 1.0f / (1.0f + expf(-b_val));
        beta_out[idx] = __float2half(beta_val);

        float a_val = __half2float(a[idx]);
        float sum_a = a_val + dt_bias[h_idx];
        float softplus_a = sum_a > 20.0f ? sum_a : logf(1.0f + expf(sum_a));
        float g_val = a_log[h_idx] * softplus_a;
        g_out[idx] = __float2half(g_val);
    }
}

// 2. Fused L2 Normalization Kernel (for Tensor Utils)
__global__ void fused_l2_normalize_f16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    float eps,
    int hidden_dim,
    int total_rows
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= total_rows) return;

    int half_dim = hidden_dim / 2;
    const half2* in_row = reinterpret_cast<const half2*>(input + row_idx * hidden_dim);
    half2* out_row = reinterpret_cast<half2*>(output + row_idx * hidden_dim);

    float sum_sq = 0.0f;
    for (int i = 0; i < half_dim; ++i) {
        float2 val = __half22float2(in_row[i]);
        sum_sq += val.x * val.x + val.y * val.y;
    }
    if (hidden_dim % 2 != 0) {
        float val = __half2float(input[row_idx * hidden_dim + hidden_dim - 1]);
        sum_sq += val * val;
    }

    float inv_norm = rsqrtf(sum_sq + eps);

    for (int i = 0; i < half_dim; ++i) {
        float2 val = __half22float2(in_row[i]);
        val.x *= inv_norm;
        val.y *= inv_norm;
        out_row[i] = __float22half2_rn(val);
    }
    if (hidden_dim % 2 != 0) {
        float val = __half2float(input[row_idx * hidden_dim + hidden_dim - 1]);
        output[row_idx * hidden_dim + hidden_dim - 1] = __float2half(val * inv_norm);
    }
}

extern "C" {
    void fused_ssm_state(
        const void* b_ptr,
        const void* a_ptr,
        const void* dt_bias_ptr,
        const void* a_log_ptr,
        void* beta_out_ptr,
        void* g_out_ptr,
        int num_v_heads,
        int total_elements
    ) {
        int total_threads = (total_elements + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;

        fused_ssm_state_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(b_ptr),
            reinterpret_cast<const half*>(a_ptr),
            reinterpret_cast<const float*>(dt_bias_ptr),
            reinterpret_cast<const float*>(a_log_ptr),
            reinterpret_cast<half*>(beta_out_ptr),
            reinterpret_cast<half*>(g_out_ptr),
            num_v_heads,
            total_elements
        );
    }

    void fused_l2_normalize(
        const void* in_ptr,
        void* out_ptr,
        float eps,
        int hidden_dim,
        int total_rows
    ) {
        int block_dim = 256;
        int grid_dim = (total_rows + block_dim - 1) / block_dim;

        fused_l2_normalize_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(in_ptr),
            reinterpret_cast<half*>(out_ptr),
            eps,
            hidden_dim,
            total_rows
        );
    }
}



// 1. Fused Attention Gate Kernel (Qwen 3.5)
__global__ void fused_attn_gate_f16_kernel(
    half* __restrict__ attn_output,
    const half* __restrict__ gate,
    int total_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    if (idx + 1 < total_elements) {
        half2 g2 = *reinterpret_cast<const half2*>(&gate[idx]);
        half2 a2 = *reinterpret_cast<const half2*>(&attn_output[idx]);

        float2 g_f2 = __half22float2(g2);
        float2 a_f2 = __half22float2(a2);

        float sig_g0 = 1.0f / (1.0f + expf(-g_f2.x));
        float sig_g1 = 1.0f / (1.0f + expf(-g_f2.y));

        float2 out_f2;
        out_f2.x = a_f2.x * sig_g0;
        out_f2.y = a_f2.y * sig_g1;

        *reinterpret_cast<half2*>(&attn_output[idx]) = __float22half2_rn(out_f2);
    } else {
        float g = __half2float(gate[idx]);
        float a = __half2float(attn_output[idx]);
        float sig_g = 1.0f / (1.0f + expf(-g));
        attn_output[idx] = __float2half(a * sig_g);
    }
}

// 2. Fused RoFormer RoPE Kernel (GLM, RoFormer)
__global__ void fused_rope_roformer_f16_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    int seq_len,
    int q_heads,
    int k_heads,
    int head_dim,
    int batch_size
) {
    int pair_idx_global = blockIdx.x * blockDim.x + threadIdx.x; 
    int half_dim = head_dim / 2;
    int total_pairs = batch_size * seq_len * q_heads * half_dim;

    if (pair_idx_global >= total_pairs) return;

    int pair_idx = pair_idx_global % half_dim;
    int head_idx = (pair_idx_global / half_dim) % q_heads;
    int seq_idx = (pair_idx_global / (half_dim * q_heads)) % seq_len;
    int b_idx = pair_idx_global / (half_dim * q_heads * seq_len);

    int q_offset1 = b_idx * (seq_len * q_heads * head_dim) + seq_idx * (q_heads * head_dim) + head_idx * head_dim + pair_idx * 2;
    int cos_offset1 = seq_idx * head_dim + pair_idx * 2;

    float2 c2 = __half22float2(*reinterpret_cast<const half2*>(&cos[cos_offset1]));
    float2 s2 = __half22float2(*reinterpret_cast<const half2*>(&sin[cos_offset1]));
    float2 q2 = __half22float2(*reinterpret_cast<const half2*>(&q[q_offset1]));

    float2 res_q;
    res_q.x = q2.x * c2.x - q2.y * s2.x;
    res_q.y = q2.y * c2.y + q2.x * s2.y;

    *reinterpret_cast<half2*>(&q[q_offset1]) = __float22half2_rn(res_q);

    if (head_idx < k_heads) {
        int k_offset1 = b_idx * (seq_len * k_heads * head_dim) + seq_idx * (k_heads * head_dim) + head_idx * head_dim + pair_idx * 2;
        float2 k2 = __half22float2(*reinterpret_cast<const half2*>(&k[k_offset1]));

        float2 res_k;
        res_k.x = k2.x * c2.x - k2.y * s2.x;
        res_k.y = k2.y * c2.y + k2.x * s2.y;

        *reinterpret_cast<half2*>(&k[k_offset1]) = __float22half2_rn(res_k);
    }
}

extern "C" {
    void fused_attn_gate(
        void* attn_output_ptr,
        const void* gate_ptr,
        int total_elements
    ) {
        int total_threads = (total_elements + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;
        fused_attn_gate_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<half*>(attn_output_ptr),
            reinterpret_cast<const half*>(gate_ptr),
            total_elements
        );
    }

    void fused_apply_rotary_pos_emb_roformer(
        void* q_ptr,
        void* k_ptr,
        const void* cos_ptr,
        const void* sin_ptr,
        int batch_size,
        int seq_len,
        int q_heads,
        int k_heads,
        int head_dim
    ) {
        int half_dim = head_dim / 2;
        int total_threads = batch_size * seq_len * q_heads * half_dim;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;

        fused_rope_roformer_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<half*>(q_ptr),
            reinterpret_cast<half*>(k_ptr),
            reinterpret_cast<const half*>(cos_ptr),
            reinterpret_cast<const half*>(sin_ptr),
            seq_len,
            q_heads,
            k_heads,
            head_dim,
            batch_size
        );
    }
}



// 3. Fused GLU / GEGLU Kernel
__global__ void fused_glu_f16_kernel(
    const half* __restrict__ input, 
    half* __restrict__ output, 
    int half_dim, 
    int total_out_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_out_elements) return;

    if (idx + 1 < total_out_elements && (idx % half_dim) + 1 < half_dim) {
        int row = idx / half_dim;
        int col = idx % half_dim;
        int in_idx1 = row * (half_dim * 2) + col;
        int in_idx2 = in_idx1 + half_dim;

        float2 x0_f2 = __half22float2(*reinterpret_cast<const half2*>(&input[in_idx1]));
        float2 x1_f2 = __half22float2(*reinterpret_cast<const half2*>(&input[in_idx2]));

        float2 res;
        res.x = x0_f2.x * (1.0f / (1.0f + expf(-x1_f2.x)));
        res.y = x0_f2.y * (1.0f / (1.0f + expf(-x1_f2.y)));

        *reinterpret_cast<half2*>(&output[idx]) = __float22half2_rn(res);
    } else {
        for(int i=0; i<2; ++i) {
            int curr = idx + i;
            if (curr >= total_out_elements) break;
            int row = curr / half_dim;
            int col = curr % half_dim;
            int in_idx1 = row * (half_dim * 2) + col;
            int in_idx2 = in_idx1 + half_dim;

            float x0 = __half2float(input[in_idx1]);
            float x1 = __half2float(input[in_idx2]);
            output[curr] = __float2half(x0 * (1.0f / (1.0f + expf(-x1))));
        }
    }
}

__global__ void fused_geglu_f16_kernel(
    const half* __restrict__ input, 
    half* __restrict__ output, 
    int half_dim, 
    int total_out_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_out_elements) return;

    if (idx + 1 < total_out_elements && (idx % half_dim) + 1 < half_dim) {
        int row = idx / half_dim;
        int col = idx % half_dim;
        int in_idx1 = row * (half_dim * 2) + col;
        int in_idx2 = in_idx1 + half_dim;

        float2 x0_f2 = __half22float2(*reinterpret_cast<const half2*>(&input[in_idx1]));
        float2 x1_f2 = __half22float2(*reinterpret_cast<const half2*>(&input[in_idx2]));

        float k0 = 0.7978845608f;
        float k1 = 0.044715f;
        float2 res;

        float inner_x = k0 * (x1_f2.x + k1 * x1_f2.x * x1_f2.x * x1_f2.x);
        res.x = 0.5f * x1_f2.x * (1.0f + tanhf(inner_x));
        res.x *= x0_f2.x;

        float inner_y = k0 * (x1_f2.y + k1 * x1_f2.y * x1_f2.y * x1_f2.y);
        res.y = 0.5f * x1_f2.y * (1.0f + tanhf(inner_y));
        res.y *= x0_f2.y;

        *reinterpret_cast<half2*>(&output[idx]) = __float22half2_rn(res);
    } else {
        for(int i=0; i<2; ++i) {
            int curr = idx + i;
            if (curr >= total_out_elements) break;
            int row = curr / half_dim;
            int col = curr % half_dim;
            int in_idx1 = row * (half_dim * 2) + col;
            int in_idx2 = in_idx1 + half_dim;

            float x0 = __half2float(input[in_idx1]);
            float x1 = __half2float(input[in_idx2]);
            float k0 = 0.7978845608f;
            float k1 = 0.044715f;
            float inner = k0 * (x1 + k1 * x1 * x1 * x1);
            output[curr] = __float2half(x0 * 0.5f * x1 * (1.0f + tanhf(inner)));
        }
    }
}

// 4. Fused Activation Kernels (QuickGELU, Mish, SoftplusStable)
__global__ void fused_quick_gelu_f16_kernel(const half* __restrict__ input, half* __restrict__ output, int total_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    if (idx + 1 < total_elements) {
        half2 in2 = *reinterpret_cast<const half2*>(&input[idx]);
        float2 x_f2 = __half22float2(in2);

        float sig0 = 1.0f / (1.0f + expf(-x_f2.x * 1.702f));
        float sig1 = 1.0f / (1.0f + expf(-x_f2.y * 1.702f));

        float2 out_f2;
        out_f2.x = x_f2.x * sig0;
        out_f2.y = x_f2.y * sig1;

        *reinterpret_cast<half2*>(&output[idx]) = __float22half2_rn(out_f2);
    } else {
        float x = __half2float(input[idx]);
        float sig = 1.0f / (1.0f + expf(-x * 1.702f));
        output[idx] = __float2half(x * sig);
    }
}

__global__ void fused_mish_f16_kernel(const half* __restrict__ input, half* __restrict__ output, int total_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    if (idx + 1 < total_elements) {
        half2 in2 = *reinterpret_cast<const half2*>(&input[idx]);
        float2 x_f2 = __half22float2(in2);

        float sp0 = x_f2.x > 20.0f ? x_f2.x : logf(1.0f + expf(x_f2.x));
        float sp1 = x_f2.y > 20.0f ? x_f2.y : logf(1.0f + expf(x_f2.y));

        float2 out_f2;
        out_f2.x = x_f2.x * tanhf(sp0);
        out_f2.y = x_f2.y * tanhf(sp1);

        *reinterpret_cast<half2*>(&output[idx]) = __float22half2_rn(out_f2);
    } else {
        float x = __half2float(input[idx]);
        float sp;
        if (x > 20.0f) {
            sp = x;
        } else {
            sp = logf(1.0f + expf(x));
        }
        float th = tanhf(sp);
        output[idx] = __float2half(x * th);
    }
}

__global__ void fused_softplus_stable_f16_kernel(const half* __restrict__ input, half* __restrict__ output, int total_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    if (idx + 1 < total_elements) {
        half2 in2 = *reinterpret_cast<const half2*>(&input[idx]);
        float2 x_f2 = __half22float2(in2);

        float x_max_0_x = fmaxf(x_f2.x, 0.0f);
        float x_max_0_y = fmaxf(x_f2.y, 0.0f);

        float sp_x = logf(1.0f + expf(-fabsf(x_f2.x))) + x_max_0_x;
        float sp_y = logf(1.0f + expf(-fabsf(x_f2.y))) + x_max_0_y;

        float2 out_f2;
        out_f2.x = sp_x;
        out_f2.y = sp_y;

        *reinterpret_cast<half2*>(&output[idx]) = __float22half2_rn(out_f2);
    } else {
        float x = __half2float(input[idx]);
        float x_max_0 = fmaxf(x, 0.0f);
        float sp = logf(1.0f + expf(-fabsf(x))) + x_max_0;
        output[idx] = __float2half(sp);
    }
}

extern "C" {
    void fused_glu(const void* in_ptr, void* out_ptr, int half_dim, int total_out_elements) {
        int block_dim = 256;
        int grid_dim = (total_out_elements + block_dim - 1) / block_dim;
        fused_glu_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), half_dim, total_out_elements);
    }
    void fused_geglu(const void* in_ptr, void* out_ptr, int half_dim, int total_out_elements) {
        int block_dim = 256;
        int grid_dim = (total_out_elements + block_dim - 1) / block_dim;
        fused_geglu_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), half_dim, total_out_elements);
    }
    void fused_quick_gelu(const void* in_ptr, void* out_ptr, int total_elements) {
        int total_threads = (total_elements + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;
        fused_quick_gelu_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), total_elements);
    }
    void fused_mish(const void* in_ptr, void* out_ptr, int total_elements) {
        int total_threads = (total_elements + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;
        fused_mish_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), total_elements);
    }
    void fused_softplus_stable(const void* in_ptr, void* out_ptr, int total_elements) {
        int total_threads = (total_elements + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;
        fused_softplus_stable_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), total_elements);
    }
}



// 5. Fused Deformable Convolution Im2Col Kernel (F32)
__global__ void fused_deform_im2col_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    const float* __restrict__ mask,
    float* __restrict__ columns,
    int in_c, int in_h, int in_w,
    int ker_h, int ker_w,
    int out_h, int out_w,
    int stride, int padding
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_kernels = in_c * out_h * out_w;
    if (index >= num_kernels) return;

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int c = index / (out_w * out_h);
    int out_c_base = c * ker_h * ker_w;

    for (int i = 0; i < ker_h; ++i) {
        for (int j = 0; j < ker_w; ++j) {
            int mask_idx = i * ker_w + j;
            int offset_idx = 2 * mask_idx;

            float mask_value = 1.0f;
            if (mask != nullptr) {
                mask_value = mask[(mask_idx * out_h + out_y) * out_w + out_x];
            }

            float offset_h = offset[(offset_idx * out_h + out_y) * out_w + out_x];
            float offset_w = offset[((offset_idx + 1) * out_h + out_y) * out_w + out_x];

            float y = (float)(out_y * stride - padding + i) + offset_h;
            float x = (float)(out_x * stride - padding + j) + offset_w;

            float val = 0.0f;
            if (y > -1.0f && y < (float)in_h && x > -1.0f && x < (float)in_w) {
                float h_low = floorf(y);
                float w_low = floorf(x);
                float h_high = h_low + 1.0f;
                float w_high = w_low + 1.0f;

                float lh = y - h_low;
                float lw = x - w_low;
                float hh = 1.0f - lh;
                float hw = 1.0f - lw;

                float w1 = hh * hw;
                float w2 = hh * lw;
                float w3 = lh * hw;
                float w4 = lh * lw;

                float v1 = (h_low >= 0.0f && w_low >= 0.0f) ? input[(c * in_h + (int)h_low) * in_w + (int)w_low] : 0.0f;
                float v2 = (h_low >= 0.0f && w_high <= in_w - 1.0f) ? input[(c * in_h + (int)h_low) * in_w + (int)w_high] : 0.0f;
                float v3 = (h_high <= in_h - 1.0f && w_low >= 0.0f) ? input[(c * in_h + (int)h_high) * in_w + (int)w_low] : 0.0f;
                float v4 = (h_high <= in_h - 1.0f && w_high <= in_w - 1.0f) ? input[(c * in_h + (int)h_high) * in_w + (int)w_high] : 0.0f;

                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
            }

            int out_row = out_c_base + i * ker_w + j;
            int out_col = out_y * out_w + out_x;
            columns[out_row * (out_h * out_w) + out_col] = mask_value * val;
        }
    }
}

// 6. Fused Z-Score Normalization Kernel
__global__ void fused_z_score_normalize_f16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int hidden_dim,
    int total_rows
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= total_rows) return;

    int half_dim = hidden_dim / 2;
    const half2* in_row = reinterpret_cast<const half2*>(input + row_idx * hidden_dim);
    half2* out_row = reinterpret_cast<half2*>(output + row_idx * hidden_dim);

    float sum = 0.0f;
    for (int i = 0; i < half_dim; ++i) {
        float2 val = __half22float2(in_row[i]);
        sum += val.x + val.y;
    }
    if (hidden_dim % 2 != 0) {
        sum += __half2float(input[row_idx * hidden_dim + hidden_dim - 1]);
    }
    float mean = sum / hidden_dim;

    float var_sum = 0.0f;
    for (int i = 0; i < half_dim; ++i) {
        float2 val = __half22float2(in_row[i]);
        float diff_x = val.x - mean;
        float diff_y = val.y - mean;
        var_sum += diff_x * diff_x + diff_y * diff_y;
    }
    if (hidden_dim % 2 != 0) {
        float diff = __half2float(input[row_idx * hidden_dim + hidden_dim - 1]) - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / hidden_dim;
    float inv_std = rsqrtf(var + 1e-5f);

    for (int i = 0; i < half_dim; ++i) {
        float2 val = __half22float2(in_row[i]);
        val.x = (val.x - mean) * inv_std;
        val.y = (val.y - mean) * inv_std;
        out_row[i] = __float22half2_rn(val);
    }
    if (hidden_dim % 2 != 0) {
        float val = __half2float(input[row_idx * hidden_dim + hidden_dim - 1]);
        output[row_idx * hidden_dim + hidden_dim - 1] = __float2half((val - mean) * inv_std);
    }
}

extern "C" {
    void fused_deform_im2col(
        const void* input_ptr,
        const void* offset_ptr,
        const void* mask_ptr,
        void* columns_ptr,
        int in_c, int in_h, int in_w,
        int ker_h, int ker_w,
        int out_h, int out_w,
        int stride, int padding
    ) {
        int num_kernels = in_c * out_h * out_w;
        int block_dim = 256;
        int grid_dim = (num_kernels + block_dim - 1) / block_dim;

        fused_deform_im2col_f32_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const float*>(input_ptr),
            reinterpret_cast<const float*>(offset_ptr),
            reinterpret_cast<const float*>(mask_ptr),
            reinterpret_cast<float*>(columns_ptr),
            in_c, in_h, in_w, ker_h, ker_w, out_h, out_w, stride, padding
        );
    }

    void fused_z_score_normalize(
        const void* in_ptr,
        void* out_ptr,
        int hidden_dim,
        int total_rows
    ) {
        int block_dim = 256;
        int grid_dim = (total_rows + block_dim - 1) / block_dim;

        fused_z_score_normalize_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(in_ptr),
            reinterpret_cast<half*>(out_ptr),
            hidden_dim,
            total_rows
        );
    }
}


// 7. Fused Recurrent Gated Delta Kernel (Qwen3.5 SSM Decoding)
__global__ void fused_recurrent_gated_delta_step_kernel(
    float* __restrict__ state, 
    const half* __restrict__ q, 
    const half* __restrict__ k, 
    const half* __restrict__ v, 
    const half* __restrict__ g, 
    const half* __restrict__ beta, 
    half* __restrict__ out, 
    int k_dim, int v_dim
) {
    int head = blockIdx.x; // batch_size * num_heads
    int v_idx = threadIdx.x; // v_dim

    if (v_idx >= v_dim) return;

    float g_val = expf(__half2float(g[head]));
    float beta_val = __half2float(beta[head]);
    float v_val = __half2float(v[head * v_dim + v_idx]);
    
    float kv_mem = 0.0f;
    
    // 1. Decay state and compute kv_mem reduction simultaneously
    for (int k_idx = 0; k_idx < k_dim; ++k_idx) {
        int state_idx = head * (k_dim * v_dim) + k_idx * v_dim + v_idx;
        float s_val = state[state_idx] * g_val;
        state[state_idx] = s_val;
        
        float k_val = __half2float(k[head * k_dim + k_idx]);
        kv_mem += s_val * k_val;
    }
    
    float delta = (v_val - kv_mem) * beta_val;
    float out_val = 0.0f;
    
    // 2. Add delta to state and compute final output reduction
    for (int k_idx = 0; k_idx < k_dim; ++k_idx) {
        int state_idx = head * (k_dim * v_dim) + k_idx * v_dim + v_idx;
        float k_val = __half2float(k[head * k_dim + k_idx]);
        
        float s_val = state[state_idx];
        s_val += k_val * delta;
        state[state_idx] = s_val;
        
        float q_val = __half2float(q[head * k_dim + k_idx]);
        out_val += s_val * q_val;
    }
    
    out[head * v_dim + v_idx] = __float2half(out_val);
}

extern "C" {
    void fused_recurrent_gated_delta_step(
        void* state_ptr,
        const void* q_ptr,
        const void* k_ptr,
        const void* v_ptr,
        const void* g_ptr,
        const void* beta_ptr,
        void* out_ptr,
        int batch_heads,
        int k_dim,
        int v_dim
    ) {
        int grid_dim = batch_heads;
        int block_dim = v_dim; 
        fused_recurrent_gated_delta_step_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<float*>(state_ptr),
            reinterpret_cast<const half*>(q_ptr),
            reinterpret_cast<const half*>(k_ptr),
            reinterpret_cast<const half*>(v_ptr),
            reinterpret_cast<const half*>(g_ptr),
            reinterpret_cast<const half*>(beta_ptr),
            reinterpret_cast<half*>(out_ptr),
            k_dim, v_dim
        );
    }
}


// 8. Fused Depthwise Conv1D Kernel (for Mamba SSM / Audio)
__global__ void fused_conv1d_depthwise_f16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int bs, int c, int len_in, int kernel_size, int len_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = bs * c * len_out;
    if (idx >= total_threads) return;

    int l_out = idx % len_out;
    int channel = (idx / len_out) % c;
    int batch = idx / (len_out * c);

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int in_idx = batch * (c * len_in) + channel * len_in + (l_out + k);
        int w_idx = channel * kernel_size + k;
        sum += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
    }

    if (bias != nullptr) {
        sum += __half2float(bias[channel]);
    }

    output[idx] = __float2half(sum);
}

// 9. Fused MROPE Cos/Sin Selector (Qwen VL)
__global__ void fused_mrope_select_f16_kernel(
    const half* __restrict__ in_all,
    half* __restrict__ out,
    int bs, int seq_len, int head_dim,
    int sec0, int sec1, int sec2
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int total = bs * seq_len * head_dim;
    if (idx >= total) return;

    int half_dim = head_dim / 2;
    if (idx + 1 < total) {
        int d_half = (idx / 2) % half_dim;
        int d = d_half * 2;
        int spatial_idx = 0;
        if (d >= sec0 && d < sec0 + sec1) {
            spatial_idx = 1;
        } else if (d >= sec0 + sec1) {
            spatial_idx = 2;
        }
        int in_idx = spatial_idx * total + idx;
        *reinterpret_cast<half2*>(&out[idx]) = *reinterpret_cast<const half2*>(&in_all[in_idx]);
    } else {
        int d = idx % head_dim;
        int spatial_idx = 0;
        if (d >= sec0 && d < sec0 + sec1) {
            spatial_idx = 1;
        } else if (d >= sec0 + sec1) {
            spatial_idx = 2;
        }
        int in_idx = spatial_idx * total + idx;
        out[idx] = in_all[in_idx];
    }
}

extern "C" {
    void fused_conv1d_depthwise(
        const void* input_ptr,
        const void* weight_ptr,
        const void* bias_ptr,
        void* out_ptr,
        int bs, int c, int len_in, int kernel_size, int len_out
    ) {
        int total = bs * c * len_out;
        int block_dim = 256;
        int grid_dim = (total + block_dim - 1) / block_dim;
        fused_conv1d_depthwise_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(input_ptr),
            reinterpret_cast<const half*>(weight_ptr),
            reinterpret_cast<const half*>(bias_ptr),
            reinterpret_cast<half*>(out_ptr),
            bs, c, len_in, kernel_size, len_out
        );
    }

    void fused_mrope_select(
        const void* in_all_ptr,
        void* out_ptr,
        int bs, int seq_len, int head_dim,
        int sec0, int sec1, int sec2
    ) {
        int total = bs * seq_len * head_dim;
        int total_threads = (total + 1) / 2;
        int block_dim = 256;
        int grid_dim = (total_threads + block_dim - 1) / block_dim;
        fused_mrope_select_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<const half*>(in_all_ptr),
            reinterpret_cast<half*>(out_ptr),
            bs, seq_len, head_dim, sec0, sec1, sec2
        );
    }
}



// 10. Fused Softmax Reduce Kernel for Flash-Decoding
__global__ void fused_softmax_reduce_f16_kernel(
    half* __restrict__ attn_weights,
    half* __restrict__ max_logits,
    half* __restrict__ sum_exp,
    int kv_len,
    int total_rows
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= total_rows) return;

    half* row_data = attn_weights + row_idx * kv_len;
    int half_len = kv_len / 2;
    half2* row_data_h2 = reinterpret_cast<half2*>(row_data);

    // 1. Find max
    float max_val = -10000.0f;
    for (int i = 0; i < half_len; ++i) {
        float2 val = __half22float2(row_data_h2[i]);
        max_val = fmaxf(max_val, fmaxf(val.x, val.y));
    }
    if (kv_len % 2 != 0) {
        float val = __half2float(row_data[kv_len - 1]);
        max_val = fmaxf(max_val, val);
    }
    max_logits[row_idx] = __float2half(max_val);

    // 2. Compute exp and sum in-place
    float sum_val = 0.0f;
    for (int i = 0; i < half_len; ++i) {
        float2 val = __half22float2(row_data_h2[i]);
        val.x = expf(val.x - max_val);
        val.y = expf(val.y - max_val);
        row_data_h2[i] = __float22half2_rn(val);
        sum_val += val.x + val.y;
    }
    if (kv_len % 2 != 0) {
        float val = __half2float(row_data[kv_len - 1]);
        float e = expf(val - max_val);
        row_data[kv_len - 1] = __float2half(e);
        sum_val += e;
    }
    sum_exp[row_idx] = __float2half(sum_val);
}

extern "C" {
    void fused_softmax_reduce(
        void* attn_weights_ptr,
        void* max_logits_ptr,
        void* sum_exp_ptr,
        int kv_len,
        int total_rows
    ) {
        int block_dim = 256;
        int grid_dim = (total_rows + block_dim - 1) / block_dim;
        fused_softmax_reduce_f16_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<half*>(attn_weights_ptr),
            reinterpret_cast<half*>(max_logits_ptr),
            reinterpret_cast<half*>(sum_exp_ptr),
            kv_len,
            total_rows
        );
    }
}