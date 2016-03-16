#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_adder(
	const MetadataType __restrict__ metadata,
	const SubGridType  __restrict__ subgrid,
	GridType           __restrict__ grid
	) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Load position in grid
    const Metadata &m = metadata[s];
    int grid_x = m.coordinate.x;
    int grid_y = m.coordinate.y;

    for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blockSize) {
        int x = i % SUBGRIDSIZE;
        int y = i / SUBGRIDSIZE;

        // Check wheter subgrid fits in grid
        if (grid_x >= 0 && grid_x < GRIDSIZE-SUBGRIDSIZE &&
            grid_y >= 0 && grid_y < GRIDSIZE-SUBGRIDSIZE) {
            // Compute shifted position in subgrid
            int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Add subgrid value to grid
            atomicAdd(&(grid[0][grid_y+y][grid_x+x]), subgrid[x][0][y_src][x_src]);
            atomicAdd(&(grid[1][grid_y+y][grid_x+x]), subgrid[x][1][y_src][x_src]);
            atomicAdd(&(grid[2][grid_y+y][grid_x+x]), subgrid[x][2][y_src][x_src]);
            atomicAdd(&(grid[3][grid_y+y][grid_x+x]), subgrid[x][3][y_src][x_src]);
        }
    }
}
}
