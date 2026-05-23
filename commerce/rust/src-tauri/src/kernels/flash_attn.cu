#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>

// 기본적인 Tiled Attention 커널 (seq_len이 크지 않을 때의 메모리 최적화 버전)
__global__ void basic_flash_attn_f16_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int seq_len,
    int head_dim,
    float scale
) {
    int batch_head_idx = blockIdx.x; 
    int q_idx = threadIdx.x; 

    if (q_idx >= seq_len) return;

    int base_offset = batch_head_idx * seq_len * head_dim;
    const half* q_row = q + base_offset + q_idx * head_dim;
    half* out_row = out + base_offset + q_idx * head_dim;

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // Attention Score 계산 (Q * K^T)을 위한 로컬 버퍼 (최대 seq_len은 제약에 맞게 할당 필요, 여기서는 동적 메모리 가정)
    extern __shared__ float s_logits[];
    float* logits = s_logits + q_idx * seq_len;

    // 1. Q * K^T
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        const half* k_row = k + base_offset + k_idx * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        score *= scale;
        logits[k_idx] = score;
        if (score > max_val) {
            max_val = score;
        }
    }

    // 2. Softmax
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        float exp_val = expf(logits[k_idx] - max_val);
        logits[k_idx] = exp_val;
        sum_exp += exp_val;
    }

    // 3. Score * V
    for (int d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            const half* v_row = v + base_offset + k_idx * head_dim;
            float prob = logits[k_idx] / sum_exp;
            out_val += prob * __half2float(v_row[d]);
        }
        out_row[d] = __float2half(out_val);
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
        // 블록당 1개의 Query 토큰을 처리 (예시: 최대 1024 토큰)
        int grid_dim = batch_size * num_heads;
        int block_dim = seq_len;
        
        // Shared memory for logits: block_dim * seq_len * sizeof(float)
        size_t shared_mem_size = block_dim * seq_len * sizeof(float);

        basic_flash_attn_f16_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_elements = batch_size * seq_len * q_heads * half_dim;

    if (idx >= total_elements) return;

    int d_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % q_heads;
    int seq_idx = (idx / (half_dim * q_heads)) % seq_len;
    int b_idx = idx / (half_dim * q_heads * seq_len);

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
    for (int i = 0; i < hidden_size; ++i) {
        float val = __half2float(xs[row_start + i]);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    
    // SiLU(x) = x / (1.0 + exp(-x))
    float silu_g = g / (1.0f + expf(-g));
    
    // Element-wise multiplication
    out[idx] = __float2half(silu_g * u);
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
        int grid_dim, block_dim;
        get_launch_config(total_elements, grid_dim, block_dim);

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int head_idx = idx % num_v_heads;

    // Beta = sigmoid(b)
    float b_val = __half2float(b[idx]);
    float beta_val = 1.0f / (1.0f + expf(-b_val));
    beta_out[idx] = __float2half(beta_val);

    // a_plus_bias = softplus(a + dt_bias) -> ln(1 + exp(x))
    float a_val = __half2float(a[idx]);
    float dt_b_val = dt_bias[head_idx];
    float sum_a = a_val + dt_b_val;
    
    // Prevent overflow in expf
    float softplus_a;
    if (sum_a > 20.0f) {
        softplus_a = sum_a;
    } else {
        softplus_a = logf(1.0f + expf(sum_a));
    }

    // g = a_log * a_plus_bias
    float alog_val = a_log[head_idx];
    float g_val = alog_val * softplus_a;
    g_out[idx] = __float2half(g_val);
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

    const half* in_row = input + row_idx * hidden_dim;
    half* out_row = output + row_idx * hidden_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float val = __half2float(in_row[i]);
        sum_sq += val * val;
    }

    float inv_norm = rsqrtf(sum_sq + eps);

    for (int i = 0; i < hidden_dim; ++i) {
        float val = __half2float(in_row[i]);
        out_row[i] = __float2half(val * inv_norm);
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
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __half2float(gate[idx]);
    float a = __half2float(attn_output[idx]);
    
    // Sigmoid: 1 / (1 + exp(-x))
    float sig_g = 1.0f / (1.0f + expf(-g));
    
    // Multiply in-place
    attn_output[idx] = __float2half(a * sig_g);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int half_dim = head_dim / 2;
    int total_pairs = batch_size * seq_len * q_heads * half_dim;

    if (idx >= total_pairs) return;

    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % q_heads;
    int seq_idx = (idx / (half_dim * q_heads)) % seq_len;
    int b_idx = idx / (half_dim * q_heads * seq_len);

    int q_offset1 = b_idx * (seq_len * q_heads * head_dim) + seq_idx * (q_heads * head_dim) + head_idx * head_dim + pair_idx * 2;
    int q_offset2 = q_offset1 + 1;

    int cos_offset1 = seq_idx * head_dim + pair_idx * 2;
    int cos_offset2 = cos_offset1 + 1;

    float c1 = __half2float(cos[cos_offset1]);
    float c2 = __half2float(cos[cos_offset2]);
    float s1 = __half2float(sin[cos_offset1]);
    float s2 = __half2float(sin[cos_offset2]);

    float q1 = __half2float(q[q_offset1]);
    float q2 = __half2float(q[q_offset2]);

    // RoFormer rotate: [-x2, x1]
    q[q_offset1] = __float2half(q1 * c1 - q2 * s1);
    q[q_offset2] = __float2half(q2 * c2 + q1 * s2);

    if (head_idx < k_heads) {
        int k_offset1 = b_idx * (seq_len * k_heads * head_dim) + seq_idx * (k_heads * head_dim) + head_idx * head_dim + pair_idx * 2;
        int k_offset2 = k_offset1 + 1;

        float k1 = __half2float(k[k_offset1]);
        float k2 = __half2float(k[k_offset2]);

        k[k_offset1] = __float2half(k1 * c1 - k2 * s1);
        k[k_offset2] = __float2half(k2 * c2 + k1 * s2);
    }
}

extern "C" {
    void fused_attn_gate(
        void* attn_output_ptr,
        const void* gate_ptr,
        int total_elements
    ) {
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out_elements) return;

    int row = idx / half_dim;
    int col = idx % half_dim;
    
    int in_idx1 = row * (half_dim * 2) + col;
    int in_idx2 = in_idx1 + half_dim;

    float x0 = __half2float(input[in_idx1]);
    float x1 = __half2float(input[in_idx2]);

    float sig_x1 = 1.0f / (1.0f + expf(-x1));
    output[idx] = __float2half(x0 * sig_x1);
}

__global__ void fused_geglu_f16_kernel(
    const half* __restrict__ input, 
    half* __restrict__ output, 
    int half_dim, 
    int total_out_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out_elements) return;

    int row = idx / half_dim;
    int col = idx % half_dim;
    
    int in_idx1 = row * (half_dim * 2) + col;
    int in_idx2 = in_idx1 + half_dim;

    float x0 = __half2float(input[in_idx1]);
    float x1 = __half2float(input[in_idx2]);

    float k0 = 0.7978845608f; // sqrt(2/pi)
    float k1 = 0.044715f;
    float inner = k0 * (x1 + k1 * x1 * x1 * x1);
    float gelu_x1 = 0.5f * x1 * (1.0f + tanhf(inner));
    
    output[idx] = __float2half(x0 * gelu_x1);
}

// 4. Fused Activation Kernels (QuickGELU, Mish, SoftplusStable)
__global__ void fused_quick_gelu_f16_kernel(const half* __restrict__ input, half* __restrict__ output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __half2float(input[idx]);
    float sig = 1.0f / (1.0f + expf(-x * 1.702f));
    output[idx] = __float2half(x * sig);
}

__global__ void fused_mish_f16_kernel(const half* __restrict__ input, half* __restrict__ output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

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

__global__ void fused_softplus_stable_f16_kernel(const half* __restrict__ input, half* __restrict__ output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __half2float(input[idx]);
    float x_max_0 = fmaxf(x, 0.0f);
    float sp = logf(1.0f + expf(-fabsf(x))) + x_max_0;
    output[idx] = __float2half(sp);
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
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;
        fused_quick_gelu_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), total_elements);
    }
    void fused_mish(const void* in_ptr, void* out_ptr, int total_elements) {
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;
        fused_mish_f16_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<const half*>(in_ptr), reinterpret_cast<half*>(out_ptr), total_elements);
    }
    void fused_softplus_stable(const void* in_ptr, void* out_ptr, int total_elements) {
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;
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

    const half* in_row = input + row_idx * hidden_dim;
    half* out_row = output + row_idx * hidden_dim;

    float sum = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        sum += __half2float(in_row[i]);
    }
    float mean = sum / hidden_dim;

    float var_sum = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float diff = __half2float(in_row[i]) - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / hidden_dim;
    float inv_std = rsqrtf(var + 1e-5f);

    for (int i = 0; i < hidden_dim; ++i) {
        float val = __half2float(in_row[i]);
        out_row[i] = __float2half((val - mean) * inv_std);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bs * seq_len * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int spatial_idx = 0;
    if (d >= sec0 && d < sec0 + sec1) {
        spatial_idx = 1;
    } else if (d >= sec0 + sec1) {
        spatial_idx = 2;
    }
    int in_idx = spatial_idx * (bs * seq_len * head_dim) + idx;
    out[idx] = in_all[in_idx];
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
        int block_dim = 256;
        int grid_dim = (total + block_dim - 1) / block_dim;
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

    // 1. Find max
    float max_val = -10000.0f;
    for (int i = 0; i < kv_len; ++i) {
        float val = __half2float(row_data[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    max_logits[row_idx] = __float2half(max_val);

    // 2. Compute exp and sum in-place
    float sum_val = 0.0f;
    for (int i = 0; i < kv_len; ++i) {
        float val = __half2float(row_data[i]);
        float e = expf(val - max_val);
        row_data[i] = __float2half(e);
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