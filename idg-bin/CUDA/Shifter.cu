#include <cuComplex.h>

#include "Types.h"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_shifter(
    unsigned jobsize,
	UVGridType __restrict__ uvgrid
	) {
    int bl = blockIdx.x;
    int x = threadIdx.x;
    int pol = threadIdx.y;
    
    // Shared data
    __shared__ float2 _uvgrid[BLOCKSIZE][BLOCKSIZE][NR_POLARIZATIONS];
    
    // Load uv grid in local memory    
    for (int y = 0; y < BLOCKSIZE; y++) {
        #if ORDER == ORDER_BL_V_U_P
        _uvgrid[y][x][pol] = uvgrid[bl][y][x][pol];
        #elif ORDER == ORDER_BL_P_V_U
        _uvgrid[y][x][pol] = uvgrid[bl][pol][y][x];
        #endif
    }
    __syncthreads();

    // Update uv grid
    #pragma unroll
    for (int y = 0; y < BLOCKSIZE; y++) {
        int x_dst = (x + (BLOCKSIZE/2)) % BLOCKSIZE;
        int y_dst = (y + (BLOCKSIZE/2)) % BLOCKSIZE;
        #if ORDER == ORDER_BL_V_U_P
        uvgrid[bl][y_dst][x_dst][pol] = _uvgrid[y][x][pol];
        #elif ORDER == ORDER_BL_P_V_U
        uvgrid[bl][pol][y_dst][x_dst] = _uvgrid[y][x][pol];
        #endif
    }
}
}
