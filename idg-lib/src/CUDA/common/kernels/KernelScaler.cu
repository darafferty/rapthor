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
    )
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Compute scaling factor
    float scale = 1 / (float(subgrid_size)*float(subgrid_size));

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;

        if (y < subgrid_size) {
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int idx = index_subgrid(subgrid_size, s, pol, y, x);
                subgrid[idx] = subgrid[idx] * scale;
            }
        }
    }
}
}
