#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {
__global__ void kernel_wtiles_to_grid(
    const unsigned int             grid_size,
    const unsigned int             tile_size,
    const unsigned int             padded_tile_size,
    const int*        __restrict__ tile_ids,
    const Coordinate* __restrict__ tile_coordinates,
    const float2*   __restrict__   tiles,
          float2*   __restrict__   grid)
{
    // Map blockIdx.x to polarizations
    unsigned int nr_polarizations = gridDim.x;
    assert(nr_polarizations <= 4);
    unsigned int pol = blockIdx.x;

    // Map blockIdx.x to tiles
    unsigned int tile_index = tile_ids[blockIdx.y];

    // Map threadIdx.x to thread id
    unsigned int tid = threadIdx.x;

    // Compute the number of threads working on one polarizaton of a tile
    unsigned int nr_threads = blockDim.x;

    // Compute the padded size of the current tile
    const Coordinate& coordinate = tile_coordinates[blockIdx.y];

    // Compute position of tile in grid
    int x0 = coordinate.x * tile_size -
             (padded_tile_size - tile_size) / 2 + grid_size / 2;
    int y0 = coordinate.y * tile_size -
             (padded_tile_size - tile_size) / 2 + grid_size / 2;
    int x_start = max(0, x0);
    int y_start = max(0, y0);

    // Add tile to grid
    for (unsigned int i = tid; i < (padded_tile_size * padded_tile_size); i += nr_threads)
    {
        unsigned int y = i / padded_tile_size;
        unsigned int x = i % padded_tile_size;

        unsigned int y_dst = y_start + y;
        unsigned int x_dst = x_start + x;

        int y_src = y_dst - y0;
        int x_src = x_dst - x0;

        if (y < padded_tile_size)
        {
            unsigned long dst_idx = index_grid_3d(grid_size, pol, y_dst, x_dst);
            unsigned long src_idx = index_grid_4d(nr_polarizations, padded_tile_size, tile_index, pol, y_src, x_src);
            atomicAdd(grid[dst_idx], tiles[src_idx]);
        }
    }
} // kernel_wtiles_to_grid
} // end extern "C"
