//! Active prefill-path Metal kernels (batch/prompt processing).
//!
//! Contains V7 GEMM, V7 residual GEMM, fused gate+up+SwiGLU GEMM,
//! batched SwiGLU, and batched RMSNorm kernels.
//!
//! Weight buffers use **SoA (Structure-of-Arrays)** layout:
//! `[all scales: total_blocks × 2 bytes][all data: total_blocks × 16 bytes]`

/// V7-based GEMM: 1D grid with weight-tiled batch processing.
///
/// Each simdgroup processes 1 weight row × ALL batch columns, loading weights
/// once per block iteration (L1 cache retains weights across columns).
/// V7 inner loop (fully unrolled, simd_sum reduction).  Input/output are
/// column-major: `inputs[col * k + elem]`, `outputs[col * n_rows + row]`.
///
/// Weight buffer uses SoA layout:
/// `[scales: n_rows*blocks_per_row × 2B][data: n_rows*blocks_per_row × 16B]`
///
/// Buffers:
/// - buffer(0) = blocks_raw  (u8, Q1_0_g128 weight data, SoA layout)
/// - buffer(1) = inputs      (f32, batch × k, column-major)
/// - buffer(2) = outputs     (f32, batch × n_rows, column-major)
/// - buffer(3) = n_rows      (u32)
/// - buffer(4) = batch_size  (u32)
/// - buffer(5) = k           (u32)
///
/// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_Q1_G128_V7: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_q1_g128_v7(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* inputs         [[buffer(1)]],
    device float* outputs              [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& batch_size          [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = n_rows * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    const uint cols = min(batch_size, 8u);

    float col_sums[8] = {0,0,0,0,0,0,0,0};

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));
        uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
        const uint inp_base = b * 32u;

        { // Chunk 0: packed.x
            uint bits = packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // Chunk 1: packed.y
            uint bits = packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // Chunk 2: packed.z
            uint bits = packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // Chunk 3: packed.w
            uint bits = packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
    }

    for (uint col = 0u; col < cols; col++) {
        float row_sum = simd_sum(col_sums[col]);
        if (lane == 0u) outputs[col * n_rows + row] = row_sum;
    }
}
"#;

/// V7-based GEMM with residual addition.
///
/// Same as `gemm_q1_g128_v7` but adds a residual value: `out = residual + gemv_result`.
///
/// Weight buffer uses SoA layout:
/// `[scales: n_rows*blocks_per_row × 2B][data: n_rows*blocks_per_row × 16B]`
///
/// Buffers:
/// - buffer(0) = blocks_raw  (u8, Q1_0_g128 weight data, SoA layout)
/// - buffer(1) = inputs      (f32, batch × k, column-major)
/// - buffer(2) = outputs     (f32, batch × n_rows, column-major)
/// - buffer(3) = n_rows      (u32)
/// - buffer(4) = batch_size  (u32)
/// - buffer(5) = k           (u32)
/// - buffer(6) = residual    (f32, batch × n_rows, column-major)
///
/// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_Q1_G128_V7_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_q1_g128_v7_residual(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* inputs         [[buffer(1)]],
    device float* outputs              [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& batch_size          [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    device const float* residual       [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = n_rows * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    const uint cols = min(batch_size, 8u);

    float col_sums[8] = {0,0,0,0,0,0,0,0};

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));
        uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
        const uint inp_base = b * 32u;

        { // Chunk 0: packed.x
            uint bits = packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // Chunk 1: packed.y
            uint bits = packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // Chunk 2: packed.z
            uint bits = packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // Chunk 3: packed.w
            uint bits = packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
    }

    for (uint col = 0u; col < cols; col++) {
        float row_sum = simd_sum(col_sums[col]);
        if (lane == 0u) outputs[col * n_rows + row] = residual[col * n_rows + row] + row_sum;
    }
}
"#;

/// Fused gate+up+SwiGLU GEMM for batch prefill (V7-based).
///
/// 1D grid with weight-tiled batch processing: each simdgroup computes one
/// FFN output position for ALL batch columns, loading gate+up weights once.
/// Applies `silu(gate) * up` in the epilogue.
///
/// Weight buffer uses SoA layout over concatenated gate+up rows
/// (total_blocks = 2*inter_size*blocks_per_row):
/// `[scales: total_blocks × 2B][data: total_blocks × 16B]`
///
/// Buffers:
/// - buffer(0) = blocks_raw  (u8, gate+up weights concatenated, SoA layout)
/// - buffer(1) = inputs      (f32, batch × k, column-major)
/// - buffer(2) = outputs     (f32, batch × inter_size, column-major)
/// - buffer(3) = inter_size  (u32)
/// - buffer(4) = batch_size  (u32)
/// - buffer(5) = k           (u32)
///
/// Dispatch: `[ceil(inter_size/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_gate_up_swiglu_gemm_q1(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* inputs         [[buffer(1)]],
    device float* outputs              [[buffer(2)]],
    constant uint& inter_size          [[buffer(3)]],
    constant uint& batch_size          [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint pos = tgid * 8u + sgid;
    if (pos >= inter_size) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = 2u * inter_size * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    const uint gate_block_base = pos * blocks_per_row;
    const uint up_block_base = (inter_size + pos) * blocks_per_row;
    const uint cols = min(batch_size, 8u);

    float gate_sums[8] = {0,0,0,0,0,0,0,0};
    float up_sums[8] = {0,0,0,0,0,0,0,0};

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint inp_base = b * 32u;

        // ── Gate: load and process 4 chunks ──
        {
            const uint gate_block_idx = gate_block_base + b;
            const float gate_scale = float(*(device const half*)(blocks_raw + gate_block_idx * 2u));
            uint4 gate_packed = *(device const uint4*)(blocks_raw + data_offset + gate_block_idx * 16u);
        { // gate chunk 0: gate_packed.x
            uint bits = gate_packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // gate chunk 1: gate_packed.y
            uint bits = gate_packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // gate chunk 2: gate_packed.z
            uint bits = gate_packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // gate chunk 3: gate_packed.w
            uint bits = gate_packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
        }

        // ── Up: load and process 4 chunks ──
        {
            const uint up_block_idx = up_block_base + b;
            const float up_scale = float(*(device const half*)(blocks_raw + up_block_idx * 2u));
            uint4 up_packed = *(device const uint4*)(blocks_raw + data_offset + up_block_idx * 16u);
        { // up chunk 0: up_packed.x
            uint bits = up_packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // up chunk 1: up_packed.y
            uint bits = up_packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // up chunk 2: up_packed.z
            uint bits = up_packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // up chunk 3: up_packed.w
            uint bits = up_packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
        }
    }

    for (uint col = 0u; col < cols; col++) {
        float gate_val = simd_sum(gate_sums[col]);
        float up_val = simd_sum(up_sums[col]);
        if (lane == 0u) {
            float silu_g = gate_val / (1.0f + exp(-gate_val));
            outputs[col * inter_size + pos] = silu_g * up_val;
        }
    }
}
"#;

/// Batched SwiGLU: processes B vectors from concatenated [gate | up] layout.
///
/// Input layout: for batch `b`, gate = gate_up[b * inter * 2 .. b * inter * 2 + inter],
///               up = gate_up[b * inter * 2 + inter .. b * inter * 2 + inter * 2].
/// Output layout: output[b * inter + elem].
///
/// Buffers:
///   - buffer(0) = gate_up `[batch_size × inter × 2]` (f32)
///   - buffer(1) = output  `[batch_size × inter]` (f32)
///   - buffer(2) = inter (u32)
///   - buffer(3) = batch_size (u32)
///
/// Dispatch: `[ceil(inter/256), batch_size, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_SWIGLU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_swiglu(
    device const float* gate_up  [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& inter         [[buffer(2)]],
    constant uint& batch_size    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint elem  = gid.x;
    uint batch = gid.y;
    if (elem >= inter || batch >= batch_size) return;

    uint offset = batch * inter * 2u;
    float g = gate_up[offset + elem];
    float u = gate_up[offset + inter + elem];
    float silu_g = g / (1.0f + exp(-g));
    output[batch * inter + elem] = silu_g * u;
}
"#;

/// Batched RMSNorm V2: one threadgroup per head, 256 threads.
///
/// Each threadgroup processes `dim` elements for a single head using
/// shared-memory parallel reduction for the sum-of-squares.
///
/// Buffers:
///   - `input`  `[num_heads × dim]` (f32)
///   - `weight` `[dim]` (f32, shared across all heads)
///   - `output` `[num_heads × dim]` (f32)
///   - `eps`    (f32 scalar)
///   - `dim`    (u32 scalar)
///
/// Dispatch: `[num_heads, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_RMSNORM_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_rmsnorm_v2(
    device const float* input,
    device const float* weight,
    device float* output,
    constant float& eps,
    constant uint& dim,
    uint tgpig  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint head = tgpig;
    uint offset = head * dim;

    threadgroup float shared_sum[256];
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = input[offset + i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tg_size) {
        output[offset + i] = input[offset + i] * rms_inv * weight[i];
    }
}
"#;
