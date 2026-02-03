#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

extern "C" __global__ void bit_serial_matmul_kernel(
    const float* input,      // [M, K]
    const uint32_t* weight,  // [N, K/32]
    const float* scales,     // [N]
    float* output,           // [M, N]
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        int k_blocks = K / 32;
        int total_dot = 0;

        for (int kb = 0; kb < k_blocks; ++kb) {
            uint32_t input_bits = 0;
            #pragma unroll
            for (int b = 0; b < 32; ++b) {
                if (input[row * K + kb * 32 + b] >= 0.0f) {
                    input_bits |= (1 << b);
                }
            }

            uint32_t xor_val = input_bits ^ weight[col * k_blocks + kb];
            total_dot += (32 - 2 * __popc(xor_val));
        }

        output[row * N + col] = (float)total_dot * scales[col];
    }
}

// Host-side wrapper to launch the kernel
extern "C" void bit_serial_matmul_kernel_wrapper(
    const float* input,
    const uint32_t* weight,
    const float* scales,
    float* output,
    int M, int N, int K
) {
    float *d_input, *d_scales, *d_output;
    uint32_t *d_weight;

    cudaMalloc(&d_input, M * K * sizeof(float));
    cudaMalloc(&d_weight, N * (K / 32) * sizeof(uint32_t));
    cudaMalloc(&d_scales, N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));

    cudaMemcpy(d_input, input, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, N * (K / 32) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bit_serial_matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_scales, d_output, M, N, K);

    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_scales);
    cudaFree(d_output);
}