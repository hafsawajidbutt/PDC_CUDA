#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 512

__global__ void scanKernel(int *d_in, int *d_out, int N) {
    __shared__ int temp[BLOCK_SIZE * 2];  // Temporary storage for the scan

    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * thid] = d_in[2 * thid];
    temp[2 * thid + 1] = d_in[2 * thid + 1];

    // Perform the scan in parallel
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Write the results to the output array
    if (thid == 0) {
        d_out[0] = temp[BLOCK_SIZE * 2 - 1];
    }
    for (int i = 1; i < BLOCK_SIZE; i++) {
        d_out[i] = temp[i];
    }
}

void prefixSum(int *d_in, int *d_out, int N) {
    int *d_in_temp, *d_out_temp;
    cudaMalloc(&d_in_temp, sizeof(int) * N);
    cudaMalloc(&d_out_temp, sizeof(int) * N);

    cudaMemcpy(d_in_temp, d_in, sizeof(int) * N, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scanKernel<<<numBlocks, BLOCK_SIZE>>>(d_in_temp, d_out_temp, N);

    cudaMemcpy(d_out, d_out_temp, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_in_temp);
    cudaFree(d_out_temp);
}

