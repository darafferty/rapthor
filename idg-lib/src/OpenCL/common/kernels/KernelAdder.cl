#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_adder(
    const int                   gridsize,
	__global const MetadataType metadata,
	__global const SubGridType  subgrid,
	__global GridType           grid
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Load position in grid
    const Metadata m = metadata[s];
    int grid_x = m.coordinate.x;
    int grid_y = m.coordinate.y;

    // Iterate all pixels in subgrid
    for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
        int y = i / SUBGRIDSIZE;
        int x = i % SUBGRIDSIZE;
        float phase = -M_PI*(x+y-SUBGRIDSIZE)/SUBGRIDSIZE;
        float2 phasor = (float2) (native_cos(phase), native_sin(phase));

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
                atomicAdd(&(grid[grid_idx]), cmul(phasor, subgrid[s][pol][y_src][x_src]));
            }
        }
    }
}
