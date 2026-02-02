#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __global__ void bit_serial_matmul_kernel(
    const float* input,      // [M, K]
    const uint32_t* weight,  // [N, K/32]
    const float* scales,     // [N]
    float* output,           // [M, N]
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Batch * Seq
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Out Features

    if (row < M && col < N) {
        int k_blocks = K / 32;
        int total_dot = 0;

        for (int kb = 0; kb < k_blocks; ++kb) {
            // 입력값의 32개 블록을 즉석에서 비트 패킹
            uint32_t input_bits = 0;
            #pragma unroll
            for (int b = 0; b < 32; ++b) {
                if (input[row * K + kb * 32 + b] >= 0.0f) {
                    input_bits |= (1 << b);
                }
            }

            // XOR + Popcount
            uint32_t xor_val = input_bits ^ weight[col * k_blocks + kb];
            total_dot += (32 - 2 * __popc(xor_val));
        }

        output[row * N + col] = (float)total_dot * scales[col];
    }
}
