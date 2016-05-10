#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_splitter(
	const MetadataType __restrict__ metadata,
	SubGridType        __restrict__ subgrid,
	const GridType     __restrict__ grid
	) {
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidx + tidy * blockDim.x;
	int blockSize = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Load position in grid
    int grid_x = metadata[s].coordinate.x;
    int grid_y = metadata[s].coordinate.y;

    // Check wheter subgrid fits in grid
    if (grid_x >= 0 && grid_x < GRIDSIZE-SUBGRIDSIZE &&
        grid_y >= 0 && grid_y < GRIDSIZE-SUBGRIDSIZE) {

        // Iterate all pixels
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blockSize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;


            // Add subgrid value to grid
            subgrid[s][0][y_dst][x_dst] = grid[0][grid_y+y][grid_x+x];
            subgrid[s][1][y_dst][x_dst] = grid[1][grid_y+y][grid_x+x];
            subgrid[s][2][y_dst][x_dst] = grid[2][grid_y+y][grid_x+x];
            subgrid[s][3][y_dst][x_dst] = grid[3][grid_y+y][grid_x+x];
        }
    }
}
}
