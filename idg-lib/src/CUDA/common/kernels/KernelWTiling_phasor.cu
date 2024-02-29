#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {
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
} // end extern "C"
