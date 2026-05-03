//! Attention Metal kernels (fused and batched attention operations).
//!
//! Contains fused QK-norm, QK-RoPE, KV-store, batched attention
//! score/softmax/weighted-sum kernels.

/// Fused QK-Norm: apply RMSNorm to both Q and K heads in a single dispatch.
///
/// The first `nq` threadgroups normalise Q heads, the remaining `nkv`
/// threadgroups normalise K heads.  Each threadgroup processes one head
/// using shared-memory parallel reduction for sum-of-squares.
///
/// Replaces two separate `batched_rmsnorm_v2` dispatches (Q-norm + K-norm).
///
/// Buffers:
///   - `q_in`     `[nq × head_dim]` (f32)
///   - `k_in`     `[nkv × head_dim]` (f32)
///   - `q_out`    `[nq × head_dim]` (f32)
///   - `k_out`    `[nkv × head_dim]` (f32)
///   - `q_weight` `[head_dim]` (f32)
///   - `k_weight` `[head_dim]` (f32)
///   - `nq`       (u32 scalar)
///   - `nkv`      (u32 scalar)
///   - `head_dim` (u32 scalar)
///   - `eps`      (f32 scalar)
///
/// Dispatch: `[nq + nkv, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_QK_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_qk_norm(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* q_weight [[buffer(4)]],
    device const float* k_weight [[buffer(5)]],
    constant uint& nq            [[buffer(6)]],
    constant uint& nkv           [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant float& eps          [[buffer(9)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    // First nq groups = Q heads, remaining nkv groups = K heads
    const bool is_q = (gid < nq);
    const uint head_idx = is_q ? gid : (gid - nq);

    device const float* in_ptr  = is_q ? (q_in + head_idx * head_dim)  : (k_in + head_idx * head_dim);
    device float* out_ptr       = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);
    device const float* w_ptr   = is_q ? q_weight : k_weight;

    // Sum of squares via shared-memory reduction
    threadgroup float shared_sum[256];
    float local_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(head_dim) + eps);

    // Apply normalization with weight
    for (uint i = tid; i < head_dim; i += tpg) {
        out_ptr[i] = in_ptr[i] * rms_inv * w_ptr[i];
    }
}
"#;

/// Fused QK-RoPE: apply rotary position embedding to both Q and K heads
/// in a single dispatch.
///
/// Thread groups are 2-D: `(ceil(half_dim/64), nq + nkv)`.  Groups with
/// `gid.y < nq` apply RoPE to Q, the rest apply to K.
///
/// Replaces two separate `batched_rope` dispatches (RoPE-Q + RoPE-K).
///
/// Buffers:
///   - `q_in`     `[nq × head_dim]` (f32)
///   - `k_in`     `[nkv × head_dim]` (f32)
///   - `q_out`    `[nq × head_dim]` (f32)
///   - `k_out`    `[nkv × head_dim]` (f32)
///   - `cos_buf`  `[half_dim]` (f32)
///   - `sin_buf`  `[half_dim]` (f32)
///   - `nq`       (u32 scalar)
///   - `nkv`      (u32 scalar)
///   - `half_dim` (u32 scalar)
///
/// Dispatch: `[ceil(half_dim/64), nq + nkv, 1]` threadgroups, `[64, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_QK_ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_qk_rope(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* cos_buf  [[buffer(4)]],
    device const float* sin_buf  [[buffer(5)]],
    constant uint& nq            [[buffer(6)]],
    constant uint& nkv           [[buffer(7)]],
    constant uint& half_dim      [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= half_dim) return;

    const bool is_q = (gid.y < nq);
    const uint head_idx = is_q ? gid.y : (gid.y - nq);
    const uint head_dim = half_dim * 2u;

    device const float* in_ptr = is_q ? (q_in + head_idx * head_dim) : (k_in + head_idx * head_dim);
    device float* out_ptr      = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);

    float c = cos_buf[d];
    float s = sin_buf[d];
    float x0 = in_ptr[d];
    float x1 = in_ptr[d + half_dim];
    out_ptr[d]            = fma(x0, c, -(x1 * s));
    out_ptr[d + half_dim] = fma(x0, s,   x1 * c);
}
"#;

/// Fused QK-Norm + QK-RoPE: apply RMSNorm then rotary position embedding
/// to both Q and K heads in a single dispatch, eliminating intermediate
/// normalised buffers.
///
/// The first `nq` threadgroups process Q heads, the remaining `nkv`
/// threadgroups process K heads.  Each threadgroup:
///   1. Computes RMSNorm via shared-memory parallel reduction.
///   2. Applies the rotary embedding in the same pass over the normalised
///      elements, writing directly to the output buffers.
///
/// Replaces two separate dispatches (`fused_qk_norm` + `fused_qk_rope`).
///
/// Buffers:
///   - `q_in`     `[nq × head_dim]`  (f32, Q slice from qkv_buf)
///   - `k_in`     `[nkv × head_dim]` (f32, K slice from qkv_buf)
///   - `q_out`    `[nq × head_dim]`  (f32, direct to q_rope_buf)
///   - `k_out`    `[nkv × head_dim]` (f32, direct to k_rope_buf)
///   - `q_weight` `[head_dim]` (f32, Q RMSNorm weights)
///   - `k_weight` `[head_dim]` (f32, K RMSNorm weights)
///   - `cos_buf`  `[half_dim]` (f32)
///   - `sin_buf`  `[half_dim]` (f32)
///   - `nq`       (u32 scalar)
///   - `nkv`      (u32 scalar)
///   - `head_dim` (u32 scalar)
///   - `eps`      (f32 scalar)
///
/// Dispatch: `[nq + nkv, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_QK_NORM_ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_qk_norm_rope(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* q_weight [[buffer(4)]],
    device const float* k_weight [[buffer(5)]],
    device const float* cos_buf  [[buffer(6)]],
    device const float* sin_buf  [[buffer(7)]],
    constant uint& nq            [[buffer(8)]],
    constant uint& nkv           [[buffer(9)]],
    constant uint& head_dim      [[buffer(10)]],
    constant float& eps          [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    const bool is_q = (gid < nq);
    const uint head_idx = is_q ? gid : (gid - nq);
    const uint half_dim = head_dim / 2u;

    device const float* in_ptr = is_q ? (q_in + head_idx * head_dim) : (k_in + head_idx * head_dim);
    device float* out_ptr      = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);
    device const float* w_ptr  = is_q ? q_weight : k_weight;

    // Phase 1: Sum of squares
    threadgroup float shared_sum[256];
    float local_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(head_dim) + eps);

    // Phase 2: Normalize + RoPE in one pass
    for (uint d = tid; d < half_dim; d += tpg) {
        float normed_lo = in_ptr[d] * rms_inv * w_ptr[d];
        float normed_hi = in_ptr[d + half_dim] * rms_inv * w_ptr[d + half_dim];

        float c = cos_buf[d];
        float s = sin_buf[d];
        out_ptr[d]            = fma(normed_lo, c, -(normed_hi * s));
        out_ptr[d + half_dim] = fma(normed_lo, s,   normed_hi * c);
    }
}
"#;

/// Fused KV-Store: copy both K and V heads into the GPU KV cache in a
/// single dispatch.
///
/// Thread groups are 2-D: `(ceil(head_dim/64), nkv)`.  Each thread copies
/// one element of one head for both K and V simultaneously.
///
/// Replaces two separate `kv_cache_store` dispatches (K-store + V-store).
///
/// Buffers:
///   - `k_data`        `[nkv × head_dim]` (f32, after RoPE)
///   - `v_data`        `[nkv × head_dim]` (f32, raw from QKV)
///   - `k_cache`       `[n_layers × nkv × max_seq × head_dim]` (f16)
///   - `v_cache`       `[n_layers × nkv × max_seq × head_dim]` (f16)
///   - `head_dim`      (u32 scalar)
///   - `nkv`           (u32 scalar)
///   - `max_seq`       (u32 scalar)
///   - `pos`           (u32 scalar)
///   - `layer_offset`  (u32 scalar, `= layer_idx * nkv * max_seq * head_dim`)
///
/// Dispatch: `[ceil(head_dim/64), nkv, 1]` threadgroups, `[64, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_KV_STORE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_kv_store(
    device const float* k_data   [[buffer(0)]],
    device const float* v_data   [[buffer(1)]],
    device half* k_cache          [[buffer(2)]],
    device half* v_cache          [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& nkv           [[buffer(5)]],
    constant uint& max_seq       [[buffer(6)]],
    constant uint& pos           [[buffer(7)]],
    constant uint& layer_offset  [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint head = gid.y;
    if (d >= head_dim || head >= nkv) return;

    const uint src_offset = head * head_dim + d;
    const uint dst_offset = layer_offset + (head * max_seq + pos) * head_dim + d;

    k_cache[dst_offset] = half(k_data[src_offset]);
    v_cache[dst_offset] = half(v_data[src_offset]);
}
"#;

/// Batched attention scores: all Q heads compute dot-product scores against
/// cached K with GQA mapping (`kv_head = q_head / heads_per_group`).
///
/// One threadgroup per (q_head, position) pair. Each threadgroup of 256
/// threads performs a parallel dot-product reduction over `head_dim`.
///
/// Buffers:
///   - `queries`            `[n_q × head_dim]` (f32)
///   - `k_cache`            `[n_kv × max_seq × head_dim]` (f32)
///   - `all_scores`         `[n_q × max_seq]` (f32, output at `q_head*max_seq+pos`)
///   - `head_dim`           (u32 scalar)
///   - `n_q`                (u32 scalar)
///   - `n_kv`               (u32 scalar)
///   - `heads_per_group`    (u32 scalar)
///   - `max_seq`            (u32 scalar)
///   - `seq_len`            (u32 scalar)
///   - `inv_sqrt_hd`        (f32 scalar)
///   - `cache_layer_offset` (u32 scalar, `= layer_idx * n_kv * max_seq * head_dim`)
///
/// Dispatch: `[n_q, seq_len, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_ATTENTION_SCORES: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_attention_scores(
    device const float* queries,
    device const half* k_cache,
    device float* all_scores,
    constant uint& head_dim,
    constant uint& n_q,
    constant uint& n_kv,
    constant uint& heads_per_group,
    constant uint& max_seq,
    constant uint& seq_len,
    constant float& inv_sqrt_hd,
    constant uint& cache_layer_offset,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntpitg [[threads_per_threadgroup]])
{
    uint q_head = tgpig.x;
    uint pos_t = tgpig.y;
    uint tid = tpitg.x;
    uint tg_size = ntpitg.x;
    if (q_head >= n_q || pos_t >= seq_len) return;

    uint kv_head = q_head / heads_per_group;

    device const float* query = queries + q_head * head_dim;
    device const half* key = k_cache + cache_layer_offset + (kv_head * max_seq + pos_t) * head_dim;

    // Parallel dot product with shared memory reduction
    threadgroup float shared[256];
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        partial = fma(query[i], float(key[i]), partial);
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        all_scores[q_head * max_seq + pos_t] = shared[0] * inv_sqrt_hd;
    }
}
"#;

/// Batched attention scores V2: reduced TG size (128) + position batching.
///
/// Each TG handles one Q head and processes `batch_stride` positions.
/// Q vector is loaded into shared memory once, reused across positions.
/// 128 threads = head_dim, so all threads are active (vs 256-thread V1 where half idle).
///
/// Grid: `[n_q, ceil(seq_len / batch_stride), 1]` TGs, `[128, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_ATTENTION_SCORES_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_attention_scores_v2(
    device const float* queries          [[buffer(0)]],
    device const half* k_cache            [[buffer(1)]],
    device float* all_scores             [[buffer(2)]],
    constant uint& head_dim              [[buffer(3)]],
    constant uint& n_q                   [[buffer(4)]],
    constant uint& n_kv                  [[buffer(5)]],
    constant uint& heads_per_group       [[buffer(6)]],
    constant uint& max_seq               [[buffer(7)]],
    constant uint& seq_len               [[buffer(8)]],
    constant float& inv_sqrt_hd          [[buffer(9)]],
    constant uint& cache_layer_offset    [[buffer(10)]],
    constant uint& batch_stride          [[buffer(11)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]])
{
    uint q_head = tgpig.x;
    uint batch_id = tgpig.y;
    if (q_head >= n_q) return;

    uint kv_head = q_head / heads_per_group;
    uint pos_start = batch_id * batch_stride;

    // Load Q vector into shared memory (reused across all positions)
    threadgroup float shared_q[128];
    if (tid < head_dim) {
        shared_q[tid] = queries[q_head * head_dim + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each position in this batch
    for (uint pos_t = pos_start; pos_t < min(pos_start + batch_stride, seq_len); pos_t++) {
        device const half* key = k_cache + cache_layer_offset + (kv_head * max_seq + pos_t) * head_dim;

        // Parallel dot product: each thread multiplies one element
        // With 128 threads and head_dim=128: exactly one element per thread
        float my_prod = 0.0f;
        if (tid < head_dim) {
            my_prod = shared_q[tid] * float(key[tid]);
        }

        // SIMD-level reduction first (fast, within simdgroup)
        float sg_sum = simd_sum(my_prod);

        // Cross-simdgroup reduction via shared memory
        // 128 threads = 4 simdgroups (32 threads each)
        threadgroup float sg_partial[4];
        uint sgid = tid / 32u;
        uint lane = tid % 32u;
        if (lane == 0u) {
            sg_partial[sgid] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float total = sg_partial[0] + sg_partial[1] + sg_partial[2] + sg_partial[3];
            all_scores[q_head * max_seq + pos_t] = total * inv_sqrt_hd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// Batched softmax: per-head numerically-stable softmax.
///
/// One threadgroup per Q head. Three-pass approach:
/// 1. Find max (parallel reduction)
/// 2. Compute exp(x - max) and accumulate sum
/// 3. Normalize by sum
///
/// Buffers:
///   - `all_scores` `[n_q × max_seq]` (f32, in-place)
///   - `n_q`        (u32 scalar)
///   - `max_seq`    (u32 scalar)
///   - `seq_len`    (u32 scalar)
///
/// Dispatch: `[n_q, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_softmax(
    device float* all_scores,
    constant uint& n_q,
    constant uint& max_seq,
    constant uint& seq_len,
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgpig >= n_q) return;

    device float* scores = all_scores + tgpig * max_seq;
    threadgroup float shared[256];

    // Pass 1: max
    float local_max = -INFINITY;
    for (uint i = tid; i < seq_len; i += tg_size) {
        local_max = max(local_max, scores[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gmax = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: exp + sum
    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += tg_size) {
        float e = exp(scores[i] - gmax);
        scores[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gsum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: normalize
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    for (uint i = tid; i < seq_len; i += tg_size) {
        scores[i] *= inv_sum;
    }
}
"#;

/// Batched attention weighted sum: per-head `output[d] = Σ_t scores[t] × V[t][d]`.
///
/// One thread per (dimension, q_head) pair with GQA mapping.
///
/// Buffers:
///   - `all_scores`         `[n_q × max_seq]` (f32)
///   - `v_cache`            `[n_kv × max_seq × head_dim]` (f16)
///   - `attn_out`           `[n_q × head_dim]` (f32, output)
///   - `head_dim`           (u32 scalar)
///   - `n_q`                (u32 scalar)
///   - `n_kv`               (u32 scalar)
///   - `heads_per_group`    (u32 scalar)
///   - `max_seq`            (u32 scalar)
///   - `seq_len`            (u32 scalar)
///   - `cache_layer_offset` (u32 scalar, `= layer_idx * n_kv * max_seq * head_dim`)
///
/// Dispatch: `[ceil(head_dim/64), n_q, 1]` threadgroups, `[64, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_ATTENTION_WEIGHTED_SUM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_attention_weighted_sum(
    device const float* all_scores,
    device const half* v_cache,
    device float* attn_out,
    constant uint& head_dim,
    constant uint& n_q,
    constant uint& n_kv,
    constant uint& heads_per_group,
    constant uint& max_seq,
    constant uint& seq_len,
    constant uint& cache_layer_offset,
    uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint q_head = gid.y;
    if (d >= head_dim || q_head >= n_q) return;

    uint kv_head = q_head / heads_per_group;
    device const float* scores = all_scores + q_head * max_seq;
    device const half* values = v_cache + cache_layer_offset + kv_head * max_seq * head_dim;

    float acc = 0.0f;
    for (uint t = 0u; t < seq_len; t++) {
        acc = fma(scores[t], float(values[t * head_dim + d]), acc);
    }
    attn_out[q_head * head_dim + d] = acc;
}
"#;
