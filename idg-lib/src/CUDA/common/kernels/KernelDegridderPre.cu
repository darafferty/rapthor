#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
    Kernel
*/
__global__ void kernel_degridder_pre(
    const unsigned               subgrid_size,
    const unsigned               nr_stations,
    const float*    __restrict__ spheroidal,
    const float2*   __restrict__ aterm,
    const Metadata* __restrict__ metadata,
          float2*   __restrict__ subgrid)
{
    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s = blockIdx.x;

	// Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int aterm_index = m.aterm_index;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;

    // Iterate all pixels in subgrid
    for (unsigned pixel = tid; pixel < subgrid_size * subgrid_size; pixel += nr_threads) {
        unsigned y = pixel / subgrid_size;
        unsigned x = pixel % subgrid_size;

        // Compute shifted position in subgrid
        int x_src = (x + (subgrid_size/2)) % subgrid_size;
        int y_src = (y + (subgrid_size/2)) % subgrid_size;

        // Load spheroidal
        float spheroidal_ = spheroidal[pixel];

        // Load pixels
        int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
        int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
        int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
        int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
        float2 pixelXX = subgrid[idx_xx] * spheroidal_;
        float2 pixelXY = subgrid[idx_xy] * spheroidal_;
        float2 pixelYX = subgrid[idx_yx] * spheroidal_;
        float2 pixelYY = subgrid[idx_yy] * spheroidal_;

        // Load aterm for station1
        float2 aXX1, aXY1, aYX1, aYY1;
        read_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, aterm, &aXX1, &aXY1, &aYX1, &aYY1);

        // Load aterm for station2
        float2 aXX2, aXY2, aYX2, aYY2;
        read_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, aterm, &aXX2, &aXY2, &aYX2, &aYY2);

        // Apply the conjugate transpose of the A-term
        apply_aterm(
            aXX1, aYX1, aXY1, aYY1,
            aXX2, aYX2, aXY2, aYY2,
            pixelXX, pixelXY, pixelYX, pixelYY);

        // Apply spheroidal and store pixels
        subgrid[idx_xx] = pixelXX;
        subgrid[idx_xy] = pixelXY;
        subgrid[idx_yx] = pixelYX;
        subgrid[idx_yy] = pixelYY;
    }
}
}
