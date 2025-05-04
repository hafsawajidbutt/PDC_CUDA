#include <stdio.h>
#include <stdlib.h>
#include "prefix_sum.cu"
#include "find_repeats.cu"

int main() {
    const int N = 1000000;  // Example size
    int *h_array = (int*) malloc(sizeof(int) * N);
    int *h_out = (int*) malloc(sizeof(int) * N);
    int *d_array, *d_repeats, *d_out;

    // Initialize array with some values
    for (int i = 0; i < N; i++) {
        h_array[i] = rand() % 100;  // Random values from 0 to 99
    }

    cudaMalloc(&d_array, sizeof(int) * N);
    cudaMalloc(&d_repeats, sizeof(int) * N);
    cudaMalloc(&d_out, sizeof(int) * N);

    cudaMemcpy(d_array, h_array, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Run prefix sum
    prefixSum(d_array, d_out, N);

    // Run find repeats
    findRepeats(d_array, d_repeats, N);

    // Output results for debugging
    printf("Prefix sum result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    printf("Repeats found:\n");
    for (int i = 0; i < N; i++) {
        if (h_repeats[i] == 1) {
            printf("Repeat found at index: %d\n", i);
        }
    }

    cudaFree(d_array);
    cudaFree(d_repeats);
    cudaFree(d_out);
    free(h_array);
    free(h_out);

    return 0;
}
