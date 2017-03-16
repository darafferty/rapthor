#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
    Kernel
*/
__global__ void kernel_scaler(
    int subgrid_size,
    float2* __restrict__ subgrid
    ) {
    // Compute scaling factor
    float scale = 1 / (float(subgrid_size)*float(subgrid_size));

    // Iterate all pixels in subgrid
    for (int y = threadIdx.y; y < subgrid_size; y += blockDim.y) {
        for (int x = threadIdx.x; x < subgrid_size; x += blockDim.x) {
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int idx = index_subgrid(subgrid_size, blockIdx.x, pol, y, x);
                float2 value = subgrid[idx];
                subgrid[idx] = make_float2(value.x * scale, value.y * scale);
            }
        }
    }
}
}
