#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void dequantize_q2_bf16_kernel(
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int num_packed_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packed_bytes) return;

    unsigned char p = packed[idx];
    
    // Logic: block_size=32. 1 byte = 4 elements. 8 bytes per scale.
    float scale = __half2float(scales[idx / 8]); 

    // Dequantize: (val - 1.5) * scale
    float v0 = (((p >> 6) & 0x03) - 1.5f) * scale;
    float v1 = (((p >> 4) & 0x03) - 1.5f) * scale;
    float v2 = (((p >> 2) & 0x03) - 1.5f) * scale;
    float v3 = ((p & 0x03) - 1.5f) * scale;

    out[idx * 4 + 0] = __float2bfloat16(v0);
    out[idx * 4 + 1] = __float2bfloat16(v1);
    out[idx * 4 + 2] = __float2bfloat16(v2);
    out[idx * 4 + 3] = __float2bfloat16(v3);
}

extern "C" void launch_dequantize_q2_bf16(
    const void* packed, const void* scales, void* out, int num_packed_bytes
) {
    int threads = 256;
    int blocks = (num_packed_bytes + threads - 1) / threads;
    dequantize_q2_bf16_kernel<<<blocks, threads>>>(
        (const unsigned char*)packed,
        (const __half*)scales,
        (__nv_bfloat16*)out,
        num_packed_bytes
    );
}

// [NEW] Fused W2A16 GEMV Kernel for Decoding (seq_len=1)
// Calculates 1 output feature per block.
extern "C" __global__ void fused_q2_gemv_bf16_kernel(
    const unsigned char* __restrict__ packed_w, // [out_features, in_features / 4]
    const __half* __restrict__ scales,          // [out_features, in_features / 32]
    const __nv_bfloat16* __restrict__ in_vec,   // [in_features]
    __nv_bfloat16* __restrict__ out_vec,        // [out_features]
    int in_features
) {
    // 1 Block = 1 Output Row
    int row = blockIdx.x; 
    int tid = threadIdx.x;
    
    // Start pointers for this row's weights and scales
    const unsigned char* row_packed = packed_w + row * (in_features / 4);
    const __half* row_scales = scales + row * (in_features / 32);

    float acc = 0.0f;

    // Each thread processes a chunk of in_features
    // Read 1 byte (4 Q2 elements), multiply with 4 in_vec elements immediately.
    for (int i = tid; i < (in_features / 4); i += blockDim.x) {
        unsigned char p = row_packed[i];
        float scale = __half2float(row_scales[i / 8]); // 1 scale per 8 bytes (32 elements)
        
        // Dequantize directly in registers
        float v0 = (((p >> 6) & 0x03) - 1.5f) * scale;
        float v1 = (((p >> 4) & 0x03) - 1.5f) * scale;
        float v2 = (((p >> 2) & 0x03) - 1.5f) * scale;
        float v3 = ((p & 0x03) - 1.5f) * scale;

        // Read input values (minimize VRAM reads)
        float in0 = __bfloat162float(in_vec[i * 4 + 0]);
        float in1 = __bfloat162float(in_vec[i * 4 + 1]);
        float in2 = __bfloat162float(in_vec[i * 4 + 2]);
        float in3 = __bfloat162float(in_vec[i * 4 + 3]);

        // Fused Multiply-Add
        acc += (v0 * in0) + (v1 * in1) + (v2 * in2) + (v3 * in3);
    }

    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(mask, acc, offset);
    }

    // Block-level reduction using Shared Memory
    static __shared__ float shared_acc[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared_acc[warp_id] = acc;
    }
    __syncthreads();

    // Warp 0 does the final sum and writes to VRAM exactly once
    if (warp_id == 0) {
        float final_acc = (lane_id < (blockDim.x / 32)) ? shared_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            final_acc += __shfl_down_sync(mask, final_acc, offset);
        }
        if (tid == 0) {
            out_vec[row] = __float2bfloat16(final_acc);
        }
    }
}

extern "C" void launch_fused_q2_gemv_bf16(
    const void* packed_w, const void* scales, const void* in_vec, void* out_vec,
    int in_features, int out_features
) {
    int threads = 128; // Adjust between 128 or 256 depending on feature size
    int blocks = out_features; // 1 block per output dimension
    fused_q2_gemv_bf16_kernel<<<blocks, threads>>>(
        (const unsigned char*)packed_w,
        (const __half*)scales,
        (const __nv_bfloat16*)in_vec,
        (__nv_bfloat16*)out_vec,
        in_features
    );
}