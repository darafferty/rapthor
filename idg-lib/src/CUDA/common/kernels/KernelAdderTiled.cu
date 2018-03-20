#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define TILE_SIZE ADDER_TILE_SIZE

extern "C" {

/*
    Kernel
*/
__global__ void kernel_adder_tiled(
    const int                    nr_subgrids,
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
    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;

    __shared__ float2 tile[TILE_SIZE*TILE_SIZE];

    // Iterate all polarizations
    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        __syncthreads();

        // Initialize tile to zero
        for (int i = tid; i < TILE_SIZE * TILE_SIZE; i++) {
            tile[i] = make_float2(0, 0);
        }

        // Iterate all subgrids
        for (int s = 0; s < nr_subgrids; s++) {
            // Load position in grid
            const Metadata &m = metadata[s];

            __syncthreads();

            // Iterate all pixels in subgrid
            for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
                int y = i / subgrid_size;
                int x = i % subgrid_size;

                // Compute shifted position in subgrid
                int x_src = (x + (subgrid_size/2)) % subgrid_size;
                int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Compute position in grid
                int grid_x = m.coordinate.x + x;
                int grid_y = m.coordinate.y + y;

                // Compute offset with respect to tile
                int tile_offset_x = grid_x - tile_x;
                int tile_offset_y = grid_y - tile_y;

                // Check whether current pixel fits in tile
                if (tile_offset_x >= 0 && tile_offset_x < TILE_SIZE &&
                    tile_offset_y >= 0 && tile_offset_y < TILE_SIZE) {

                    // Compute phasor
                    float phase = M_PI*(x+y-subgrid_size)/subgrid_size;
                    float2 phasor = make_float2(cos(phase), sin(phase));

                    // Update tile
                    long dst_idx = tile_offset_y * TILE_SIZE + tile_offset_x;
                    long src_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                    tile[dst_idx] += phasor * subgrid[src_idx];
                }

            } // end for i (pixels)
        } // end for s (subgrids)

        __syncthreads();

        // Write tile to grid
        for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += nr_threads) {
            // Compute position in tile
            int y = i / TILE_SIZE;
            int x = i % TILE_SIZE;

            // Compute position in grid
            int grid_y = tile_y + y;
            int grid_x = tile_x + x;

            // Check whether current pixel fits in grid
            if (grid_x < grid_size && grid_y < grid_size) {
                // Update grid
                long grid_idx = index_grid(grid_size, pol, tile_y + y, tile_x + x);
                grid[grid_idx] += tile[i];
            }
        } // end for i (pixels)
    } // end for pol (correlations)
} // end kernel_adder
}
