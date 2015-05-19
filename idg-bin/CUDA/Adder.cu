#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_adder(
    int jobsize,
    int bl_offset,
	const CoordinateType	__restrict__ coordinates,
	const UVGridType	    __restrict__ uvgrid,
	GridType         		__restrict__ grid
	) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	int bl = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = x + y * blockDim.x;

    // Load coordinate
    __shared__ Coordinate _coordinate;
    if (tid == 0) {
        _coordinate = coordinates[bl + bl_offset];
    }
    __syncthreads();
    
    // Compute thread position in grid
    int grid_x = _coordinate.x + x;
    int grid_y = _coordinate.y + y;
    
    // Add uvgrid point to grid
    #if ORDER == ORDER_BL_P_V_U
    atomicAdd(&(grid[0][grid_y][grid_x]), uvgrid[bl][0][y][x]);
    atomicAdd(&(grid[1][grid_y][grid_x]), uvgrid[bl][1][y][x]);
    atomicAdd(&(grid[2][grid_y][grid_x]), uvgrid[bl][2][y][x]);
    atomicAdd(&(grid[3][grid_y][grid_x]), uvgrid[bl][3][y][x]);
    #elif ORDER == ORDER_BL_V_U_P
    atomicAdd(&(grid[0][grid_y][grid_x]), uvgrid[bl][y][x][0]);
    atomicAdd(&(grid[1][grid_y][grid_x]), uvgrid[bl][y][x][1]);
    atomicAdd(&(grid[2][grid_y][grid_x]), uvgrid[bl][y][x][2]);
    atomicAdd(&(grid[3][grid_y][grid_x]), uvgrid[bl][y][x][3]);
    #endif
}
}
