#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// return GB/sec
float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

// CUDA kernel function to perform SAXPY operation
__global__ void saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        result[index] = alpha * x[index] + y[index];
}

void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {
    int totalBytes = sizeof(float) * 3 * N;

    // Number of threads per block and number of blocks
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Pointers for device memory
    float *device_x = nullptr, *device_y = nullptr, *device_result = nullptr;

    // Allocate device memory
    cudaMalloc(&device_x, N * sizeof(float));
    cudaMalloc(&device_y, N * sizeof(float));
    cudaMalloc(&device_result, N * sizeof(float));

    // Copy input arrays from host to device
    cudaMemcpy(device_x, xarray, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, N * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
    double startTime = CycleTimer::currentSeconds();

    // Launch CUDA kernel
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    // Check for any CUDA errors
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(errCode));
    }

    // Copy result from device to host
    cudaMemcpy(resultarray, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // End timing
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
