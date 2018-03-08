#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
    Kernel
*/
__global__ void kernel_adder(
    const long                   grid_size,
    const int                    subgrid_size,
    const Metadata* __restrict__ metadata,
    const float2*   __restrict__ subgrid,
          float2*   __restrict__ grid)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Load position in grid
    const Metadata &m = metadata[s];
    int grid_x = m.coordinate.x;
    int grid_y = m.coordinate.y;

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;
        float phase = M_PI*(x+y-subgrid_size)/subgrid_size;
        float2 phasor = make_float2(cos(phase), sin(phase));

        // Check wheter subgrid fits in grid
        if (grid_x >= 0 && grid_x < grid_size-subgrid_size &&
            grid_y >= 0 && grid_y < grid_size-subgrid_size) {
            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Add subgrid value to grid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                long dst_idx = index_grid(grid_size, pol, grid_y + y, grid_x + x);
                long src_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                atomicAdd(&(grid[dst_idx]), phasor * subgrid[src_idx]);
            }
        }
    }
} // end kernel_adder
}
