//! Utility Metal kernels (activations, normalization, element-wise ops).
//!
//! Contains RMSNorm, SwiGLU, softmax, ReLU, SiLU, residual-add,
//! matrix-vector multiply, and argmax kernels.

/// Numerically-stable softmax.
///
/// Buffers: `"x"` → input (0), `"result"` → output (1)
/// Scalars: `"n"` → size (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& size        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    float max_val = input[0];
    for (uint i = 1u; i < size; i++) {
        max_val = max(max_val, input[i]);
    }

    float my_exp = exp(input[gid] - max_val);

    float sum_exp = 0.0f;
    for (uint i = 0u; i < size; i++) {
        sum_exp += exp(input[i] - max_val);
    }

    output[gid] = (sum_exp > 0.0f) ? (my_exp / sum_exp) : (1.0f / float(size));
}
"#;

/// Element-wise ReLU: y = max(0, x).
///
/// Buffers: `"x"` → input (0), `"result"` → output (1)
/// Scalars: `"n"` → count (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = max(0.0f, input[gid]);
}
"#;

/// RMSNorm: y_i = x_i / sqrt(mean(x²) + eps) * weight_i
///
/// Buffers: `"x"` → input (0), `"y"` → weight (1), `"result"` → output (2)
/// Scalars: `"alpha"` → eps (3), `"n"` → count (4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RMSNORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    constant uint& n           [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    float sum_sq = 0.0f;
    for (uint i = 0u; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = rsqrt(sum_sq / float(n) + eps);

    output[gid] = input[gid] * rms * weight[gid];
}
"#;

/// SiLU (Sigmoid Linear Unit): y = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Buffers: `"x"` → input (0), `"result"` → output (1)
/// Scalars: `"n"` → count (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SILU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}
"#;

/// SwiGLU fused activation: `output[i] = silu(gate[i]) * up[i]`
///
/// Buffers: `"x"` → gate (0), `"y"` → up (1), `"result"` → output (2)
/// Scalars: `"n"` → count (3)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SWIGLU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void swiglu(
    device const float* gate    [[buffer(0)]],
    device const float* up      [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * up[gid];
}
"#;

/// Residual add in-place: `a[i] += b[i]`
///
/// Buffer `"x"` is read-write (both input and output).
///
/// Buffers: `"x"` → a (0, read-write), `"y"` → b (1)
/// Scalars: `"n"` → count (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RESIDUAL_ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void residual_add(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    a[gid] += b[gid];
}
"#;

/// Fused SwiGLU reading from concatenated [gate, up] buffer.
///
/// `gate = buffer[0..n]`, `up = buffer[n..2n]`
/// `output[i] = silu(gate[i]) * up[i]`
///
/// Buffers: `"x"` → gate_up (0), `"result"` → output (1)
/// Scalars: `"n"` → n (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SWIGLU_FUSED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void swiglu_fused(
    device const float* gate_up  [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate_up[gid];
    float u = gate_up[n + gid];
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * u;
}
"#;

/// RMSNorm with weight (weighted variant for LLM layers).
///
/// Computes `output[i] = (input[i] / sqrt(mean(input²) + eps)) * weight[i]`.
/// Each thread redundantly computes the full sum-of-squares — correct for
/// typical hidden sizes (e.g. 4096) and avoids shared-memory reduction.
///
/// Buffers: `"x"` → input (0), `"y"` → weight (1), `"result"` → output (2)
/// Scalars: `"n"` → count (3), `"alpha"` → eps (4)
///
/// RMSNorm with weight vector, for use from scirs2-core dispatch.
///
/// **Scalar binding order**: scirs2-core binds scalars in a fixed order:
/// `"alpha"`, `"beta"`, `"n"`, `"m"`, `"k"` — only those that are set.
/// Since we `set_f32("alpha", eps)` and `set_u32("n", h)`, the binding is:
///   buffer(3) = alpha (eps, f32), buffer(4) = n (h, u32).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RMSNORM_WEIGHTED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_weighted(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    float sum_sq = 0.0f;
    for (uint i = 0u; i < n; i++) {
        float v = input[i];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / float(n) + eps);

    output[gid] = input[gid] * rms * weight[gid];
}
"#;

/// Optimized RMSNorm with parallel threadgroup reduction.
///
/// Fixes the O(n²) issue in V1 where every thread redundantly computes
/// the full sum-of-squares. V2 uses cooperative threadgroup reduction:
///
/// 1. Each of 256 threads computes a partial sum of `x²` over strided elements
/// 2. Tree reduction in threadgroup shared memory to get total sum
/// 3. All threads compute `rms = rsqrt(sum/n + eps)` from shared result
/// 4. All threads apply `output[i] = input[i] * rms * weight[i]`
///
/// Complexity: O(n) total work instead of O(n²).
///
/// Buffers: `"x"` → input (0), `"y"` → weight (1), `"result"` → output (2)
/// Scalars: `"alpha"` → eps (3), `"n"` → count (4)
///
/// Dispatch: `[1, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RMSNORM_WEIGHTED_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_weighted_v2(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_sum[256];

    // Step 1: Each thread computes partial sum of squares
    float partial_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float v = input[i];
        partial_sum = fma(v, v, partial_sum);
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Tree reduction in shared memory
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Compute rms scaling factor (all threads read same value)
    float rms = rsqrt(shared_sum[0] / float(n) + eps);

    // Step 4: Apply scaling to output
    for (uint i = tid; i < n; i += tg_size) {
        output[i] = input[i] * rms * weight[i];
    }
}
"#;

/// FP32 matrix-vector multiply: y = A * x
///
/// Buffers: `"x"` → matrix a (0), `"y"` → vector x (1), `"result"` → output (2)
/// Scalars: `"n"` → m (3), `"k"` → k (4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_MATVEC_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matvec_f32(
    device const float* a      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& m           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= m) return;

    float sum = 0.0f;
    uint row_offset = gid * k;
    for (uint j = 0u; j < k; j++) {
        sum += a[row_offset + j] * x[j];
    }
    output[gid] = sum;
}
"#;

/// GPU argmax — finds the index of the maximum value in a float array.
///
/// Uses a single threadgroup with 1024 threads (sufficient for vocab ≤ ~500K).
/// Each thread scans every 1024th element, then a tree reduction finds the
/// global maximum's index.
///
/// Buffers:
/// - buffer(0) = data    (f32, input values)
/// - buffer(1) = result  (uint32, output index — single element)
/// - buffer(2) = count   (uint32, scalar)
///
/// Dispatch: `[1, 1, 1]` threadgroups, `[1024, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_ARGMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax(
    device const float* data    [[buffer(0)]],
    device uint* result         [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    threadgroup float shared_vals[1024];
    threadgroup uint shared_idxs[1024];

    float best_val = -INFINITY;
    uint best_idx = 0u;

    for (uint i = tid; i < count; i += tpg) {
        float v = data[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        result[0] = shared_idxs[0];
    }
}
"#;
