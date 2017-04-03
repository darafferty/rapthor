#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_splitter(
    const int                grid_size,
	__global const Metadata* metadata,
	__global float2*         subgrid,
	__global const float2*   grid)
{
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
    if (grid_x >= 0 && grid_x < grid_size-SUBGRIDSIZE &&
        grid_y >= 0 && grid_y < grid_size-SUBGRIDSIZE) {

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int x = i % SUBGRIDSIZE;
            int y = i / SUBGRIDSIZE;
            float phase = -M_PI*(x+y-SUBGRIDSIZE)/SUBGRIDSIZE;
            float2 phasor = (float2) (native_cos(phase), native_sin(phase));

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Set grid value to subgrid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int src_idx = index_grid(grid_size, pol, grid_y + y, grid_x + x);
                int dst_idx = index_subgrid(SUBGRIDSIZE, s, pol, y_dst, x_dst);
                subgrid[dst_idx] = cmul(phasor, grid[src_idx]);
            }
        }
    }
}
