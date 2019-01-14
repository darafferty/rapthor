#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
    Kernel
*/
__global__ void kernel_splitter(
    const long                   grid_size,
    const int                    subgrid_size,
    const Metadata* __restrict__ metadata,
    float2*         __restrict__ subgrid,
    const float2*   __restrict__ grid,
    const bool                   enable_tiling)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Load position in grid
    int grid_x = metadata[s].coordinate.x;
    int grid_y = metadata[s].coordinate.y;

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
        int x = i % subgrid_size;
        int y = i / subgrid_size;
        float pi = (float) M_PI;
        float phase = -pi*(x+y-subgrid_size)/subgrid_size;
        float2 phasor = make_float2(cos(phase), sin(phase));

        // Check whether subgrid fits in grid
        if (grid_x >= 0 && grid_x < grid_size-subgrid_size &&
            grid_y >= 0 && grid_y < grid_size-subgrid_size) {

            // Compute shifted position in subgrid
            int x_dst = (x + (subgrid_size/2)) % subgrid_size;
            int y_dst = (y + (subgrid_size/2)) % subgrid_size;

            // Set grid value to subgrid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                long src_idx = enable_tiling ?
                    index_grid_tiled(grid_size, pol, grid_y + y, grid_x + x) :
                    index_grid(grid_size, pol, grid_y + y, grid_x + x);
                long dst_idx = index_subgrid(subgrid_size, s, pol, y_dst, x_dst);
                subgrid[dst_idx] = phasor * grid[src_idx];
            }
        }
    }
}
} // end kernel_splitter
