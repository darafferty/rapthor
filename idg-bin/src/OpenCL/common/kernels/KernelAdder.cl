#include "math.cl"

#include "Types.cl"

#define TILE_FACTOR ADDER_TILE_FACTOR

/*
	Kernel
*/
__kernel void kernel_adder(
    const int                   nr_subgrids,
    const int                   gridsize,
	__global const MetadataType metadata,
	__global const SubGridType  subgrid,
	__global GridType           grid
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);
    int gidx = get_group_id(0);
    int gidy = get_group_id(1);
    int tilesize = gridsize / TILE_FACTOR;

    // Compute tile position
    int tile_x = gidx * tilesize;
    int tile_y = gidy * tilesize;

    for (int s = 0; s < nr_subgrids; s++) {
        // Load position in grid
        const Metadata m = metadata[s];
        int grid_x = m.coordinate.x;
        int grid_y = m.coordinate.y;

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            if (grid_x+x >= tile_x && grid_x+x < tile_x+tilesize &&
                grid_y+y >= tile_y && grid_y+y < tile_y+tilesize) {

                // Compute shifted position in subgrid
                int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Add subgrid value to grid
                #pragma unroll 4
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    int grid_idx = (pol * gridsize * gridsize) + ((grid_y + y) * gridsize) + (grid_x + x);
                    grid[grid_idx] += subgrid[s][pol][y_src][x_src];
                }
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
