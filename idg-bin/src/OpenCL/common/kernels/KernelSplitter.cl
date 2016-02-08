#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_splitter(
	__global const MetadataType metadata,
	__global SubGridType        subgrid,
	__global const GridType     grid
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
        int x = tid % SUBGRIDSIZE;
        int y = tid / SUBGRIDSIZE;

        // Load position in grid
        const Metadata m = metadata[s];
        int grid_x = m.coordinate.x;
        int grid_y = m.coordinate.y;

        // Check wheter subgrid fits in grid
        if (grid_x >= 0 && grid_x < GRIDSIZE-SUBGRIDSIZE &&
            grid_y >= 0 && grid_y < GRIDSIZE-SUBGRIDSIZE) {
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
