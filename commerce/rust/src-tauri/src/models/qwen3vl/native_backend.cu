#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

extern "C" __global__ void bit_serial_matmul_kernel(
    const float* input,      // [M, K] - Already on GPU
    const uint32_t* weight,  // [N, K/32] - Already on GPU
    const float* scales,     // [N] - Already on GPU
    float* output,           // [M, N] - Already on GPU
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        int k_blocks = K / 32;
        int total_dot = 0;

        for (int kb = 0; kb < k_blocks; ++kb) {
            uint32_t input_bits = 0;
            // [OPTIMIZATION] Fast bit packing from input
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

// Low-level wrapper for pre-allocated GPU memory
extern "C" void bit_serial_matmul_cuda_direct(
    const float* d_input,
    const uint32_t* d_weight,
    const float* d_scales,
    float* d_output,
    int M, int N, int K,
    int device_id
) {
    cudaSetDevice(device_id);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bit_serial_matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_scales, d_output, M, N, K);
}

// Simple wrapper for one-off calls (Backward compatibility)
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

    bit_serial_matmul_cuda_direct(d_input, d_weight, d_scales, d_output, M, N, K, 0);

    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_scales);
    cudaFree(d_output);
}
