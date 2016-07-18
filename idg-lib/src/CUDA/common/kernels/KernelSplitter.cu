#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_splitter(
    const int                       gridsize,
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

    // Check whether subgrid fits in grid
    if (grid_x >= 0 && grid_x < gridsize-SUBGRIDSIZE &&
        grid_y >= 0 && grid_y < gridsize-SUBGRIDSIZE) {

        // Iterate all pixels
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blockSize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Set grid value to subgrid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int grid_idx = (pol * gridsize * gridsize) + ((grid_y + y) * gridsize) + (grid_x + x);
                subgrid[s][pol][y_dst][x_dst] = grid[grid_idx];
            }
        }
    }
}
}
