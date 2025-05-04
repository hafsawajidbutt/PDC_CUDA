#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "cudaRenderer.h"
#include "circleRenderer.h"
#include "util.h"

//------------------------------------------------------------------------------
// Kernel to render circles
//   One CUDA thread per pixel.  Each thread:
//    1) Loads the background color,
//    2) Loops over all circles, blends each circle's color contribution into
//       the pixel based on its position and radius,
//    3) Writes the final color back to the pixel.
//------------------------------------------------------------------------------
__global__ void kernelRenderPixels() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int W = cuConstRendererParams.imageWidth;
    int H = cuConstRendererParams.imageHeight;
    
    if (x >= W || y >= H) return;

    // 1) Load the cleared-background color
    int pixIdx = y * W + x;
    int off = 4 * pixIdx;
    float4 accum = *(float4*)(&cuConstRendererParams.imageData[off]);

    // normalized pixel center
    float2 pc = make_float2((x + 0.5f) / W, (y + 0.5f) / H);

    // pull in circle buffers & scene info
    int N = cuConstRendererParams.numCircles;
    float* P = cuConstRendererParams.position;  // x, y, z triples
    float* R = cuConstRendererParams.radius;    // per-circle radius
    float* C = cuConstRendererParams.color;     // rgb triples
    auto sc = cuConstRendererParams.sceneName;

    // 2) Blend each circle in input order
    for (int i = 0; i < N; ++i) {
        // read center & radius
        float3 ctr = *(float3*)(&P[3 * i]);
        float rad = R[i];
        float2 d;
        d.x = ctr.x - pc.x;
        d.y = ctr.y - pc.y;
        
        if (d.x * d.x + d.y * d.y > rad * rad) continue;

        float3 rgb;
        float alpha;
        if (sc == SNOWFLAKES || sc == SNOWFLAKES_SINGLE_FRAME) {
            // reuse your snowflake shading math
            float dist = sqrtf(d.x * d.x + d.y * d.y);
            float norm = dist / rad;
            rgb = lookupColor(norm);  // from lookupColor.cu_inl
            float kMax = .6f + .4f * (1.f - ctr.z);
            float c = fmaxf(0.f, fminf(kMax, 1.f));
            alpha = .5f * expf(-4.f * norm * norm) * c;
        } else {
            // simple half-opaque circles
            rgb.x = C[3 * i + 0];
            rgb.y = C[3 * i + 1];
            rgb.z = C[3 * i + 2];
            alpha = .5f;
        }

        float invA = 1.f - alpha;
        accum.x = alpha * rgb.x + invA * accum.x;
        accum.y = alpha * rgb.y + invA * accum.y;
        accum.z = alpha * rgb.z + invA * accum.z;
        accum.w = alpha + accum.w;
    }

    // 3) Write back exactly once
    *(float4*)(&cuConstRendererParams.imageData[off]) = accum;
}

//------------------------------------------------------------------------------
// Render the scene
//------------------------------------------------------------------------------
void CudaRenderer::render() {
    // Launch one thread per pixel in a 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y
    );
    
    kernelRenderPixels<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}
