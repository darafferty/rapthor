/*
    Kernel
*/
__kernel void kernel_splitter(
    const int                grid_size,
    const int                subgrid_size,
    __global const Metadata* metadata,
    __global float2*         subgrid,
    __global const float2*   grid)
{
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int nr_threads = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Load position in grid
    const Metadata m = metadata[s];
    int grid_x = m.coordinate.x;
    int grid_y = m.coordinate.y;

    // Check whether subgrid fits in grid
    if (grid_x >= 0 && grid_x < grid_size-subgrid_size &&
        grid_y >= 0 && grid_y < grid_size-subgrid_size) {

        // Iterate all pixels in subgrid
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int x = i % subgrid_size;
            int y = i / subgrid_size;
            float phase = -M_PI*(x+y-subgrid_size)/subgrid_size;
            float2 phasor = (float2) (native_cos(phase), native_sin(phase));

            // Compute shifted position in subgrid
            int x_dst = (x + (subgrid_size/2)) % subgrid_size;
            int y_dst = (y + (subgrid_size/2)) % subgrid_size;

            // Set grid value to subgrid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int src_idx = index_grid(grid_size, pol, grid_y + y, grid_x + x);
                int dst_idx = index_subgrid(subgrid_size, s, pol, y_dst, x_dst);
                subgrid[dst_idx] = cmul(phasor, grid[src_idx]);
            }
        }
    }
} // end kernel_splitter
