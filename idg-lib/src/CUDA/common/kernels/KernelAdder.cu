#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_adder(
    const int                       gridsize,
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
        int y = i / SUBGRIDSIZE;
        int x = i % SUBGRIDSIZE;
        float phase = -M_PI*(x+y-SUBGRIDSIZE)/SUBGRIDSIZE;
        float2 phasor = make_float2(cos(phase), sin(phase));

        // Check wheter subgrid fits in grid
        if (grid_x >= 0 && grid_x < gridsize-SUBGRIDSIZE &&
            grid_y >= 0 && grid_y < gridsize-SUBGRIDSIZE) {
            // Compute shifted position in subgrid
            int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Add subgrid value to grid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int grid_idx = (pol * gridsize * gridsize) + ((grid_y + y) * gridsize) + (grid_x + x);
                atomicAdd(&(grid[grid_idx]), phasor * subgrid[s][pol][y_src][x_src]);
            }
        }
    }
}
}
