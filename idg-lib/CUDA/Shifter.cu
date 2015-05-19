#include <cuComplex.h>

#include "Types.h"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_shifter(
    int jobsize,
	SubGridType __restrict__ subgrid
	) {
    int bl = blockIdx.x;
    int x = threadIdx.x;
    int pol = threadIdx.y;
    
    // Shared data
    __shared__ float2 _subgrid[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
    
    for (int chunk = 0; chunk < CHUNKSIZE; chunk++) {
        // Load uv grid in local memory    
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            #if ORDER == ORDER_BL_V_U_P
            _subgrid[y][x][pol] = subgrid[bl][chunk][y][x][pol];
            #elif ORDER == ORDER_BL_P_V_U
            _subgrid[y][x][pol] = subgrid[bl][chunk][pol][y][x];
            #endif
        }
        __syncthreads();

        // Update uv grid
        #pragma unroll
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            #if ORDER == ORDER_BL_V_U_P
            subgrid[bl][chunk][y_dst][x_dst][pol] = _subgrid[y][x][pol];
            #elif ORDER == ORDER_BL_P_V_U
            subgrid[bl][chunk][pol][y_dst][x_dst] = _subgrid[y][x][pol];
            #endif
        }
    }
}
}
