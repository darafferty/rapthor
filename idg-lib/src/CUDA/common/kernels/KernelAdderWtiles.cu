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
    assert(gridDim.x == NR_POLARIZATIONS);
    unsigned int pol = blockIdx.x;

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
                size_t dst_idx = index_grid(dst_tile_size, dst_tile_index, pol, y, x);
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
            size_t dst_idx = index_grid(dst_tile_size, dst_tile_index, pol, dst_y, dst_x);
            size_t src_idx = index_grid(src_tile_size, src_tile_index, pol, src_y, src_x);
            dst_tiles[dst_idx] = src_tiles[src_idx];
            src_tiles[src_idx] = make_float2(0, 0);
        }
    }
} // end kernel_copy_tiles

__global__ void kernel_apply_phasor(
    const float                    image_size,
    const float                    w_step,
    const int                      w_padded_tile_size,
          float2*     __restrict__ w_padded_tiles,
    const float*      __restrict__ shift,
    const Coordinate* __restrict__ tile_coordinates)
{
    // Map blockIdx.x to polarizations
    assert(gridDim.x == NR_POLARIZATIONS);
    unsigned int pol = blockIdx.x;

    // Map blockIdx.y to tile_index
    unsigned int tile_index = blockIdx.y;

    // Map threadIdx.x to thread id
    unsigned int tid = threadIdx.x;

    // Compute the number of threads working on one polarizaton of a tile
    unsigned int nr_threads = blockDim.x;

    // Compute cell_size
    float cell_size = image_size / w_padded_tile_size;

    // Compute scale
    float scale = 1.0f / (w_padded_tile_size * w_padded_tile_size);

    // Compute W
    const Coordinate& coordinate = tile_coordinates[tile_index];
    float w = (coordinate.z + 0.5f) * w_step;

    for (int i = tid; i < (w_padded_tile_size * w_padded_tile_size); i += nr_threads)
    {
        int y = i / w_padded_tile_size;
        int x = i % w_padded_tile_size;

        if (y < w_padded_tile_size) {
            // Compute phase
            const int x_ = (x + (w_padded_tile_size / 2)) % w_padded_tile_size;
            const int y_ = (y + (w_padded_tile_size / 2)) % w_padded_tile_size;

            // Use alternative computation of n to work around accuracy issues
            const float l = (x_ - (w_padded_tile_size / 2)) * cell_size - shift[0];
            const float m = (y_ - (w_padded_tile_size / 2)) * cell_size - shift[1];
            const float n = 1.0f - sqrtf(1.0 - (l * l) - (m * m));

            const float pi = (float) M_PI;
            const float phase = -2 * pi * n * w;

            // Compute phasor
            float2 phasor = make_float2(cosf(phase), sinf(phase)) * scale;

            // Apply correction
            size_t idx = index_grid(w_padded_tile_size, tile_index, pol, y, x);
            w_padded_tiles[idx] = (w_padded_tiles[idx] * phasor);
        }
    }
} // end kernel_apply_phasor

__global__ void kernel_subgrids_to_wtiles(
    const long                   grid_size,
    const int                    subgrid_size,
    const int                    wtile_size,
    const int                    subgrid_offset,
    const Metadata* __restrict__ metadata,
    const float2*   __restrict__ subgrid,
          float2*   __restrict__ padded_tiles,
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
    int tile_top = m.wtile_coordinate.x * wtile_size -
                    subgrid_size / 2 + grid_size / 2;
    int tile_left = m.wtile_coordinate.y * wtile_size -
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
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                long dst_idx = index_grid(wtile_size + subgrid_size, tile_index, pol, y_dst, x_dst);
                long src_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                float2 value = phasor * subgrid[src_idx];
                atomicAdd(padded_tiles[dst_idx], value);
            }
        }
    }
}

__global__ void kernel_wtiles_to_grid(
    const unsigned int             tile_size,
    const unsigned int             padded_tile_size,
    const unsigned int             grid_size,
    const int*        __restrict__ tile_ids,
    const Coordinate* __restrict__ tile_coordinates,
    const float2*   __restrict__   padded_tiles,
          float2*   __restrict__   grid)
{
    // Map blockIdx.x to polarizations
    assert(gridDim.x == NR_POLARIZATIONS);
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
    int x0 = coordinate.x * tile_size - (padded_tile_size - tile_size) / 2 +
             grid_size / 2;
    int y0 = coordinate.y * tile_size - (padded_tile_size - tile_size) / 2 +
             grid_size / 2;
    int x_start = max(0, x0);
    int y_start = max(0, y0);

    // Add tile to grid
    for (unsigned int i = tid; i < (padded_tile_size * padded_tile_size); i += nr_threads)
    {
        unsigned int y = i / padded_tile_size;
        unsigned int x = i % padded_tile_size;

        unsigned int y_dst = y_start + y;
        unsigned int x_dst = x_start + x;

        unsigned int y_src = y_dst - y0;
        unsigned int x_src = x_dst - x0;

        if (y < padded_tile_size)
        {
            unsigned long dst_idx = index_grid(grid_size, pol, y_dst, x_dst);
            unsigned long src_idx = index_grid(padded_tile_size, tile_index, pol, y_src, x_src);
            atomicAdd(grid[dst_idx], padded_tiles[src_idx]);
        }
    }
}

} // end extern "C"
