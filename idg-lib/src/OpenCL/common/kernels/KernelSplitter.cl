#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_splitter(
    const int                   gridsize,
	__global const MetadataType metadata,
	__global SubGridType        subgrid,
	__global const GridType     grid
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

    // Check wheter subgrid fits in grid
    if (grid_x >= 0 && grid_x < gridsize-SUBGRIDSIZE &&
        grid_y >= 0 && grid_y < gridsize-SUBGRIDSIZE) {

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int x = i % SUBGRIDSIZE;
            int y = i / SUBGRIDSIZE;

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
