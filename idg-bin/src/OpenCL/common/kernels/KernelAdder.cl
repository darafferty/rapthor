#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_adder(
    const int nr_subgrids,
	__global const MetadataType metadata,
	__global const SubGridType  subgrid,
	__global GridType           grid
	) {
    // Determine position in grid
    int nr_groups_x = get_num_groups(0);
    int nr_groups_y = get_num_groups(1);
    int blocksize_x = GRIDSIZE / get_num_groups(0);
    int blocksize_y = GRIDSIZE / get_num_groups(1);
    int block_x_first = get_group_id(0) * blocksize_x;
    int block_y_first = get_group_id(1) * blocksize_y;
    int block_x_last = block_x_first + blocksize_x;
    int block_y_last = block_y_first + blocksize_x;

    // Deterime position in group
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);

    // Iterate all subgrids
    for (int s = 0; s < nr_subgrids; s++) {

        // Load position in grid
        const Metadata m = metadata[s];
        int grid_x = m.coordinate.x;
        int grid_y = m.coordinate.y;

        // Skip subgrids that are completely out of range
        if (grid_x < block_x_first - SUBGRIDSIZE ||
            grid_y < block_y_first - SUBGRIDSIZE ||
            grid_x > block_x_last ||
            grid_y > block_y_last) {
            continue;
        }

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int x = i % SUBGRIDSIZE;
            int y = i / SUBGRIDSIZE;

            // Compute position n grid
            int x_dst = grid_x + x;
            int y_dst = grid_y + y;

            // Skip pixels that are out of range
            if (x_dst < block_x_first ||
                x_dst > block_x_last  ||
                y_dst < block_y_first ||
                y_dst > block_y_last) {
                continue;
            }

            // Compute shifted position in subgrid
            int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Add subgrid value to grid
            grid[0][y_dst][x_dst] += subgrid[s][0][y_src][x_src];
            grid[1][y_dst][x_dst] += subgrid[s][1][y_src][x_src];
            grid[2][y_dst][x_dst] += subgrid[s][2][y_src][x_src];
            grid[3][y_dst][x_dst] += subgrid[s][3][y_src][x_src];
        }
    }
}
