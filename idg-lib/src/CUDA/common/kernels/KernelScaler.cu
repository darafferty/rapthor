#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_scaler(
    int subgrid_size,
	SubGridType __restrict__ subgrid
	) {
    // Compute scaling factor
    float scale = 1 / (float(SUBGRIDSIZE)*float(SUBGRIDSIZE));

	// Iterate all pixels in subgrid
	for (int y = threadIdx.y; y < SUBGRIDSIZE; y += blockDim.y) {
		for (int x = threadIdx.x; x < SUBGRIDSIZE; x += blockDim.x) {
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                float2 value = subgrid[blockIdx.x][pol][y][x];
                subgrid[blockIdx.x][pol][y][x] = make_float2(value.x * scale, value.y * scale);
            }
        }
    }
}
}
