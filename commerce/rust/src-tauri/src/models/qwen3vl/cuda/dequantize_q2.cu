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
