#include "Types.h"
#include "math.cu"

#include "KernelCalibrate_index.cuh"

extern "C" {

__global__ void kernel_calibrate_lmnp(
    const int                    grid_size,
    const int                    subgrid_size,
    const float                  image_size,
    const float                  w_step,
    const Metadata* __restrict__ metadata,
          float4*   __restrict__ lmnp)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const Coordinate coordinate = m.coordinate;

    // Location of current subgrid
    const int x_coordinate = coordinate.x;
    const int y_coordinate = coordinate.y;
    const int z_coordinate = coordinate.z;

    for (unsigned int i = tid; i < nr_pixels; i += nr_threads) {
        unsigned int y = i / subgrid_size;
        unsigned int x = i % subgrid_size;

        if (y < subgrid_size) {

            // Compute u,v,w offset in wavelenghts
            const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
            const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
            const float w_offset = w_step * ((float) z_coordinate + 0.5) * 2 * M_PI;

            // Compute l,m,n and phase_offset
            float l = compute_l(x, subgrid_size, image_size);
            float m = compute_m(y, subgrid_size, image_size);
            float n = compute_n(l, m);
            float phase_offset = u_offset*l + v_offset*m + w_offset*n;

            // Store result
            unsigned int lmnp_idx = index_lmnp(subgrid_size, s, y, x);
            lmnp[lmnp_idx] = make_float4(l, m, n, phase_offset);
        }
    } // end for i
} // end kernel_calibrate_lmnp

} // end extern "C"
