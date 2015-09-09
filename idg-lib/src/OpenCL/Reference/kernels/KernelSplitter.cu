#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_splitter(
    int jobsize,
	const MetadataType __restrict__ metadata,
	SubGridType        __restrict__ subgrid,
	const GridType     __restrict__ grid
	) {
	int tid = threadIdx.x;
	int s = blockIdx.x + blockIdx.y * gridDim.x;

    // Extract all subgrids from grid
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
            
            // Add subgrid point to grid
            subgrid[s][0][y][x] = grid[0][grid_y][grid_x];
            subgrid[s][1][y][x] = grid[1][grid_y][grid_x];
            subgrid[s][2][y][x] = grid[2][grid_y][grid_x];
            subgrid[s][3][y][x] = grid[3][grid_y][grid_x];
        }

        __syncthreads();
    }
}
}
