#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {
__global__ void kernel_wtiles_to_patch(
    const unsigned int             nr_tiles,
    const long                     grid_size,
    const unsigned int             tile_size,
    const unsigned int             padded_tile_size,
    const unsigned int             patch_size,
    const Coordinate               patch_coordinate,
    const int*        __restrict__ tile_ids,
    const Coordinate* __restrict__ tile_coordinates,
    const float2*     __restrict__ tiles,
    float2*           __restrict__ patch)
{
    // Map blockIdx.x to polarization
    unsigned int nr_polarizations = gridDim.x;
    assert(nr_polarizations <= 4);
    unsigned int pol = blockIdx.x;

    // Map blockIdx.y to row of patch
    unsigned int y = blockIdx.y;

    // Map threadIdx.x to thread id
    unsigned int tid = threadIdx.x;

    // Compute the number of threads working on one polarization/row of a patch
    unsigned int nr_threads = blockDim.x;

    for (unsigned int i = 0; i < nr_tiles; i++)
    {
        unsigned int tile_index = tile_ids[i];

        // Compute position of tile in grid
        const Coordinate& coordinate = tile_coordinates[tile_index];
        int x0 = coordinate.x * tile_size -
                 (padded_tile_size - tile_size) / 2 + grid_size / 2;
        int y0 = coordinate.y * tile_size -
                 (padded_tile_size - tile_size) / 2 + grid_size / 2;
        int x_start = x0;
        int y_start = y0;
        int x_end = x0 + padded_tile_size;
        int y_end = y0 + padded_tile_size;

        // Shift start to inside patch
        x_start = max(x_start, patch_coordinate.x);
        y_start = max(y_start, patch_coordinate.y);

        // Shift end to inside patch
        x_end = min(x_end, patch_coordinate.x + patch_size);
        y_end = min(y_end, patch_coordinate.y + patch_size);

        // Compute number of pixels to process
        int height = y_end - y_start;
        int width  = x_end - x_start;

        // Compute y position in patch and tile
        unsigned int y_patch = y_start + y - patch_coordinate.y;
        unsigned int y_tile  = y_start + y - y0;

        // Add tile to patch
        if (y < height && y_patch < patch_size && width > 0)
        {
            for (unsigned int x = tid; x < width; x += nr_threads)
            {
                // Compute x position in patch and tile
                unsigned int x_patch = x_start + x - patch_coordinate.x;
                unsigned int x_tile  = x_start + x - x0;

                // Add tile value to patch
                unsigned long idx_patch = index_grid_3d(patch_size, pol, y_patch, x_patch);
                unsigned long idx_tile  = index_grid_4d(nr_polarizations, padded_tile_size, tile_index, pol, y_tile, x_tile);
                atomicAdd(patch[idx_patch], tiles[idx_tile]);
            }
        } // end if y
    } // end for i
} // end kernel_wtiles_to_patch
} // end extern "C"
