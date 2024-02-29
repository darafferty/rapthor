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
} // end extern "C"
