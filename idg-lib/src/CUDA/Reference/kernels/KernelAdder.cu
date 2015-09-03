#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_adder(
    int jobsize,
	const MetadataType __restrict__ metadata,
	const SubGridType  __restrict__ subgrid,
	GridType           __restrict__ grid
	) {
    int tid = threadIdx.x;
	int s = blockIdx.x + blockIdx.y * gridDim.x;

    // Add all subgrids to grid
    for (; s < jobsize; s += gridDim.x) {
	    // Load metadata
        const Metadata m = metadata[s];
        int x_coordinate = m.coordinate.x;
        int y_coordinate = m.coordinate.y;

        for (int pixel = tid; pixel < SUBGRIDSIZE * SUBGRIDSIZE; pixel += blockDim.x) {
            // Compute x and y position
            int x = pixel % SUBGRIDSIZE;
            int y = pixel / SUBGRIDSIZE;

            // Compute thread position in grid
            int grid_x = x_coordinate + x;
            int grid_y = y_coordinate + y;
            
            // Add subgrid pixel to grid
            atomicAdd(&(grid[0][grid_y][grid_x]), subgrid[s][0][y][x]);
            atomicAdd(&(grid[1][grid_y][grid_x]), subgrid[s][1][y][x]);
            atomicAdd(&(grid[2][grid_y][grid_x]), subgrid[s][2][y][x]);
            atomicAdd(&(grid[3][grid_y][grid_x]), subgrid[s][3][y][x]);
        }

        __syncthreads();
    }
}
}
