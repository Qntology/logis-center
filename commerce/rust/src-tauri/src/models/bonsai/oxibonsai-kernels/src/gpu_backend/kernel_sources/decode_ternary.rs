//! TQ2_0_g128 Metal GEMV kernel (decode path).

/// TQ2_0_g128 GEMV V1 — SIMD-group-per-row, SoA weight layout.
///
/// SoA layout: `[all d: N×2 bytes FP16 LE][all qs: N×32 bytes]`
/// Encoding: 0b00→-1, 0b01→0, 0b10→+1, 0b11→0 (4 weights/byte, LSB-first)
/// Dispatch: 256 threads, `[ceil(n_rows/8), 1, 1]` threadgroups.
/// Buffers: "x"→SoA weights(0), "y"→input float4*(1), "result"→output(2)
/// Scalars: "n"→n_rows(3), "k"→k(4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_TQ2_G128_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float decode_tq2(uint code) {
    return select(select(0.0f, -1.0f, code == 0u), 1.0f, code == 2u);
}

inline float4 decode_byte_tq2(uint b) {
    return float4(
        decode_tq2((b     ) & 3u),
        decode_tq2((b >> 2) & 3u),
        decode_tq2((b >> 4) & 3u),
        decode_tq2((b >> 6) & 3u)
    );
}

kernel void gemv_tq2_g128_v1(
    device const uchar*  soa_raw   [[buffer(0)]],
    device const float4* input4    [[buffer(1)]],
    device       float*  output    [[buffer(2)]],
    constant uint&       n_rows    [[buffer(3)]],
    constant uint&       k         [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(soa_raw + block_idx * 2u));
        const uint qs_base = qs_offset + block_idx * 32u;
        uint w0 = (uint)(soa_raw[qs_base +  0]) | ((uint)(soa_raw[qs_base +  1]) << 8) | ((uint)(soa_raw[qs_base +  2]) << 16) | ((uint)(soa_raw[qs_base +  3]) << 24);
        uint w1 = (uint)(soa_raw[qs_base +  4]) | ((uint)(soa_raw[qs_base +  5]) << 8) | ((uint)(soa_raw[qs_base +  6]) << 16) | ((uint)(soa_raw[qs_base +  7]) << 24);
        uint w2 = (uint)(soa_raw[qs_base +  8]) | ((uint)(soa_raw[qs_base +  9]) << 8) | ((uint)(soa_raw[qs_base + 10]) << 16) | ((uint)(soa_raw[qs_base + 11]) << 24);
        uint w3 = (uint)(soa_raw[qs_base + 12]) | ((uint)(soa_raw[qs_base + 13]) << 8) | ((uint)(soa_raw[qs_base + 14]) << 16) | ((uint)(soa_raw[qs_base + 15]) << 24);
        uint w4 = (uint)(soa_raw[qs_base + 16]) | ((uint)(soa_raw[qs_base + 17]) << 8) | ((uint)(soa_raw[qs_base + 18]) << 16) | ((uint)(soa_raw[qs_base + 19]) << 24);
        uint w5 = (uint)(soa_raw[qs_base + 20]) | ((uint)(soa_raw[qs_base + 21]) << 8) | ((uint)(soa_raw[qs_base + 22]) << 16) | ((uint)(soa_raw[qs_base + 23]) << 24);
        uint w6 = (uint)(soa_raw[qs_base + 24]) | ((uint)(soa_raw[qs_base + 25]) << 8) | ((uint)(soa_raw[qs_base + 26]) << 16) | ((uint)(soa_raw[qs_base + 27]) << 24);
        uint w7 = (uint)(soa_raw[qs_base + 28]) | ((uint)(soa_raw[qs_base + 29]) << 8) | ((uint)(soa_raw[qs_base + 30]) << 16) | ((uint)(soa_raw[qs_base + 31]) << 24);
        const uint inp_base = b * 32u;
        float block_sum = 0.0f;
        { uint w = w0;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 0u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 1u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 2u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 3u]); }
        { uint w = w1;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 4u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 5u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 6u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 7u]); }
        { uint w = w2;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 8u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 9u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 10u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 11u]); }
        { uint w = w3;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 12u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 13u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 14u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 15u]); }
        { uint w = w4;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 16u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 17u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 18u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 19u]); }
        { uint w = w5;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 20u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 21u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 22u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 23u]); }
        { uint w = w6;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 24u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 25u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 26u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 27u]); }
        { uint w = w7;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 28u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 29u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 30u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 31u]); }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;
