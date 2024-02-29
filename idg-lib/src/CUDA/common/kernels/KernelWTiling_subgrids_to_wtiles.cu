#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {
__global__ void kernel_subgrids_to_wtiles(
    const int                    nr_polarizations,
    const long                   grid_size,
    const int                    subgrid_size,
    const int                    tile_size,
    const int                    subgrid_offset,
    const Metadata* __restrict__ metadata,
    const float2*   __restrict__ subgrid,
          float2*   __restrict__ tiles,
          float2                 scale)
{
    // Map blockIdx.x to subgrids
    int s = blockIdx.x + subgrid_offset;

    // Map thread indices to thread id
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;

    // Compute the number of threads working on one subgrid
    int nr_threads = blockDim.x * blockDim.y;

    // Load tile coordinates
    const Metadata &m = metadata[s];
    int tile_index = m.wtile_index;
    int tile_top = m.wtile_coordinate.x * tile_size -
                    subgrid_size / 2 + grid_size / 2;
    int tile_left = m.wtile_coordinate.y * tile_size -
                    subgrid_size / 2 + grid_size / 2;

    // Compute position in tile
    int subgrid_x = m.coordinate.x - tile_top;
    int subgrid_y = m.coordinate.y - tile_left;

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;
        float pi = (float) M_PI;
        float phase = pi * (x+y-subgrid_size)/subgrid_size;
        float2 phasor = make_float2(cosf(phase) * scale.x,
                                    sinf(phase) * scale.y);

        if (y < subgrid_size)
        {
            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Compute position in grid
            int x_dst = subgrid_x + x;
            int y_dst = subgrid_y + y;

            // Add subgrid value to grid
            #pragma unroll 4
            for (int pol = 0; pol < nr_polarizations; pol++) {
                long dst_idx = index_grid_4d(nr_polarizations, tile_size + subgrid_size, tile_index, pol, y_dst, x_dst);
                long src_idx = index_subgrid(nr_polarizations, subgrid_size, s, pol, y_src, x_src);
                float2 value = phasor * subgrid[src_idx];
                atomicAdd(tiles[dst_idx], value);
            }
        }
    }
} // kernel_subgrids_to_wtiles
} // end extern "C"
