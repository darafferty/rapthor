#include "Types.h"
#include "math.cu"

extern "C" {

/*
    Kernel
*/
__global__ void kernel_calibrate(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         nr_channels,
    const int                         nr_terms,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    int s          = blockIdx.x;
    int tidx       = threadIdx.x;
    int tidy       = threadIdx.y;
    int tid        = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;

    // Load metadata for first subgrid
    const Metadata &m_0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
} // end kernel_calibrate

} // end extern "C"
