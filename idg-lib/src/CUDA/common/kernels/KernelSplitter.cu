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
    const Metadata &m = metadata[s];
    int subgrid_x= m.coordinate.x;
    int subgrid_y= m.coordinate.y;
    int subgrid_w = m.coordinate.z;
    bool negative_w = subgrid_w < 0;

    // Determine polarization index
    const int index_pol_default[NR_POLARIZATIONS]    = {0, 1, 2, 3};
    const int index_pol_transposed[NR_POLARIZATIONS] = {0, 2, 1, 3};
    int *index_pol = (int *) (negative_w ? index_pol_default : index_pol_transposed);

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
        int x = i % subgrid_size;
        int y = i / subgrid_size;
        float pi = (float) M_PI;
        float phase = -pi*(x+y-subgrid_size)/subgrid_size;
        float2 phasor = make_float2(cos(phase), sin(phase));

        // Check whether subgrid fits in grid
        if (subgrid_x >= 1 && subgrid_x < grid_size-subgrid_size &&
            subgrid_y >= 1 && subgrid_y < grid_size-subgrid_size) {

            // Compute shifted position in subgrid
            int x_dst = (x + (subgrid_size/2)) % subgrid_size;
            int y_dst = (y + (subgrid_size/2)) % subgrid_size;

            // Compute position in grid
            int x_src = negative_w ? grid_size - subgrid_x - x : subgrid_x + x;
            int y_src = negative_w ? grid_size - subgrid_y - y : subgrid_y + y;

            // Set grid value to subgrid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int pol_src = index_pol[pol];
                long src_idx = enable_tiling ?
                    index_grid_tiling(TILE_SIZE_GRID, grid_size, pol_src, y_src, x_src) :
                    index_grid(grid_size, pol_src, y_src, x_src);
                long dst_idx = index_subgrid(subgrid_size, s, pol, y_dst, x_dst);
                float2 value = grid[src_idx];
                value = negative_w ? conj(value) : value;
                subgrid[dst_idx] = phasor * value;
            }
        }
    }
}
} // end kernel_splitter
