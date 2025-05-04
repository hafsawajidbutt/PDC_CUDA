#include <cuda_runtime.h>
#include <stdio.h>

__global__ void findRepeatsKernel(int *d_array, int *d_repeats, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        for (int i = idx + 1; i < N; i++) {
            if (d_array[idx] == d_array[i]) {
                d_repeats[idx] = 1;  // Mark as a repeat
                d_repeats[i] = 1;    // Mark as a repeat
            }
        }
    }
}

void findRepeats(int *d_array, int *d_repeats, int N) {
    int *d_array_temp, *d_repeats_temp;
    cudaMalloc(&d_array_temp, sizeof(int) * N);
    cudaMalloc(&d_repeats_temp, sizeof(int) * N);

    cudaMemcpy(d_array_temp, d_array, sizeof(int) * N, cudaMemcpyHostToDevice);

    int numBlocks = (N + 512 - 1) / 512;
    findRepeatsKernel<<<numBlocks, 512>>>(d_array_temp, d_repeats_temp, N);

    cudaMemcpy(d_repeats, d_repeats_temp, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_array_temp);
    cudaFree(d_repeats_temp);
}
