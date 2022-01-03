// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {
__global__ void kernel_copy_tiles(
    const unsigned int             src_tile_size,
    const unsigned int             dst_tile_size,
    const int*        __restrict__ src_tile_ids,
    const int*        __restrict__ dst_tile_ids,
          float2*     __restrict__ src_tiles,
          float2*     __restrict__ dst_tiles)
{
    // Map blockIdx.x to polarizations
    unsigned int nr_polarizations = gridDim.x;
    assert(nr_polarizations <= 4);
    unsigned int pol = blockIdx.x;

    // Tranpose the polarizations
    const int index_pol_transposed[4] = {0, 2, 1, 3};
    unsigned int src_pol = pol;
    unsigned int dst_pol = index_pol_transposed[pol];

    // Map blockIdx.y to tile_id
    unsigned int src_tile_index = src_tile_ids[blockIdx.y];
    unsigned int dst_tile_index = dst_tile_ids[blockIdx.y];

    // Map threadIdx.x to thread id
    unsigned int tid = threadIdx.x;

    // Compute the number of threads working on one polarizaton of a tile
    unsigned int nr_threads = blockDim.x;

    // Compute remaining parameters
    int padding = dst_tile_size - src_tile_size;
    int copy_tile_size = min(src_tile_size, dst_tile_size);

    // Reset dst_tile to zero if src_tile is smaller
    if (padding > 0)
    {
        for (unsigned int i = tid; i < (dst_tile_size * dst_tile_size); i += nr_threads)
        {
            unsigned int y = i / dst_tile_size;
            unsigned int x = i % dst_tile_size;

            if (y < dst_tile_size)
            {
                size_t dst_idx = index_grid_4d(nr_polarizations, dst_tile_size, dst_tile_index, pol, y, x);
                dst_tiles[dst_idx] = make_float2(0, 0);
            }
        }
    }

    __syncthreads();

    // Copy src_tile to dst_tile and reset src_tile to zero
    for (unsigned int i = tid; i < (copy_tile_size * copy_tile_size); i += nr_threads)
    {
        unsigned int src_y = i / copy_tile_size;
        unsigned int src_x = i % copy_tile_size;
        unsigned int dst_y = src_y;
        unsigned int dst_x = src_x;

        if (padding > 0) {
            dst_y += padding / 2;
            dst_x += padding / 2;
        } else if (padding < 0) {
            src_y -= padding / 2;
            src_x -= padding / 2;
        }

        if (src_y < src_tile_size && dst_y < dst_tile_size)
        {
            size_t dst_idx = index_grid_4d(nr_polarizations, dst_tile_size, dst_tile_index, dst_pol, dst_y, dst_x);
            size_t src_idx = index_grid_4d(nr_polarizations, src_tile_size, src_tile_index, src_pol, src_y, src_x);
            dst_tiles[dst_idx] = src_tiles[src_idx];
            src_tiles[src_idx] = make_float2(0, 0);
        }
    }
} // end kernel_copy_tiles

__global__ void kernel_apply_phasor(
    const float                    image_size,
    const float                    w_step,
    const int                      tile_size,
          float2*     __restrict__ tiles,
    const float*      __restrict__ shift,
    const Coordinate* __restrict__ tile_coordinates,
    const int                      sign)
{
    // Map blockIdx.x to polarizations
    unsigned int nr_polarizations = gridDim.x;
    assert(nr_polarizations <= 4);
    unsigned int pol = blockIdx.x;

    // Map blockIdx.y to tile_index
    unsigned int tile_index = blockIdx.y;

    // Map threadIdx.x to thread id
    unsigned int tid = threadIdx.x;

    // Compute the number of threads working on one polarizaton of a tile
    unsigned int nr_threads = blockDim.x;

    // Compute cell_size
    float cell_size = image_size / tile_size;

    // Compute scale
    float scale = 1.0f / (tile_size * tile_size);

    // Compute W
    const Coordinate& coordinate = tile_coordinates[tile_index];
    float w = (coordinate.z + 0.5f) * w_step;

    for (int i = tid; i < (tile_size * tile_size); i += nr_threads)
    {
        int y = i / tile_size;
        int x = i % tile_size;

        if (y < tile_size) {
            // Inline FFT shift
            int x_ = (x + (tile_size / 2)) % tile_size;
            int y_ = (y + (tile_size / 2)) % tile_size;

            // Compute phase
            const float l = (x_ - (tile_size / 2)) * cell_size;
            const float m = (y_ - (tile_size / 2)) * cell_size;
            const float n = compute_n(l, -m, shift);
            const float phase = sign * 2 * M_PI * n * w;

            // Compute phasor
            float2 phasor = make_float2(cosf(phase), sinf(phase)) * scale;

            // Apply correction
            size_t idx = index_grid_4d(nr_polarizations, tile_size, tile_index, pol, y, x);
            tiles[idx] = tiles[idx] * phasor;
        }
    }
} // end kernel_apply_phasor

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

__global__ void kernel_subgrids_from_wtiles(
    const int                    nr_polarizations,
    const long                   grid_size,
    const int                    subgrid_size,
    const int                    tile_size,
    const int                    subgrid_offset,
    const Metadata* __restrict__ metadata,
          float2*   __restrict__ subgrid,
    const float2*   __restrict__ tiles)
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
        float phase = -pi * (x+y-subgrid_size)/subgrid_size;
        float2 phasor = make_float2(cosf(phase), sinf(phase));

        if (y < subgrid_size)
        {
            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Compute position in grid
            int x_dst = subgrid_x + x;
            int y_dst = subgrid_y + y;

            // Set subgrid value from grid
            #pragma unroll 4
            for (int pol = 0; pol < nr_polarizations; pol++) {
                long src_idx = index_grid_4d(nr_polarizations, tile_size + subgrid_size, tile_index, pol, y_dst, x_dst);
                long dst_idx = index_subgrid(nr_polarizations, subgrid_size, s, pol, y_src, x_src);
                subgrid[dst_idx] = tiles[src_idx] * phasor;
            }
        }
    }
} // kernel_subgrids_from_grid

__global__ void kernel_wtiles_from_grid(
    const unsigned int             grid_size,
    const unsigned int             tile_size,
    const unsigned int             padded_tile_size,
    const int*        __restrict__ tile_ids,
    const Coordinate* __restrict__ tile_coordinates,
          float2*     __restrict__ tiles,
    const float2*     __restrict__ grid)
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

    // Extract tile from grid
    for (unsigned int i = tid; i < (padded_tile_size * padded_tile_size); i += nr_threads)
    {
        unsigned int y = i / padded_tile_size;
        unsigned int x = i % padded_tile_size;

        unsigned int y_src = y_start + y;
        unsigned int x_src = x_start + x;

        int y_dst = y_src - y0;
        int x_dst = x_src - x0;

        if (y < padded_tile_size)
        {
            unsigned long src_idx = index_grid_3d(grid_size, pol, y_src, x_src);
            unsigned long dst_idx = index_grid_4d(nr_polarizations, padded_tile_size, tile_index, pol, y_dst, x_dst);
            tiles[dst_idx] = grid[src_idx];
        }
    }
} // kernel_wtiles_from_grid

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

__global__ void kernel_wtiles_from_patch(
    const unsigned int             nr_tiles,
    const long                     grid_size,
    const unsigned int             tile_size,
    const unsigned int             padded_tile_size,
    const unsigned int             patch_size,
    const Coordinate               patch_coordinate,
    const int*        __restrict__ tile_ids,
    const Coordinate* __restrict__ tile_coordinates,
          float2*     __restrict__ tiles,
    const float2*     __restrict__ patch)
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
        int x_end = x_start + padded_tile_size;
        int y_end = y_start + padded_tile_size;

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

        // Set tile from patch
        if (y < height && y_patch < patch_size && width > 0)
        {
            for (unsigned int x = tid; x < width; x += nr_threads)
            {
                // Compute y position in patch and tile
                unsigned int x_patch = x_start + x - patch_coordinate.x;
                unsigned int x_tile  = x_start + x - x0;

                // Set tile value from patch
                unsigned long idx_patch = index_grid_3d(patch_size, pol, y_patch, x_patch);
                unsigned long idx_tile  = index_grid_4d(nr_polarizations, padded_tile_size, tile_index, pol, y_tile, x_tile);
                tiles[idx_tile] = patch[idx_patch];
            }
        } // end if y
    } // end for i
} // end kernel_wtiles_from_patch

} // end extern "C"
