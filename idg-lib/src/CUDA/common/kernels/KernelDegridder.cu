#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

__shared__ float4 shared[3][BATCH_SIZE];

/*
    Kernel
*/
template<int current_nr_channels>
__device__ void kernel_degridder_(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         nr_channels,
    const int                         channel_offset,
    const int                         nr_stations,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
          float2*        __restrict__ visibilities,
    const float*         __restrict__ spheroidal,
    const float2*        __restrict__ aterms,
    const int*           __restrict__ aterms_indices,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ subgrid)
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
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) m.coordinate.z + 0.5) * 2 * M_PI;

    // Iterate timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        int aterm_idx_current = aterms_indices[time_offset_global + time_offset_local];

        // Determine number of timesteps to process
        current_nr_timesteps = 0;
        for (int time = time_offset_local; time < nr_timesteps; time++) {
            if (aterms_indices[time_offset_global + time] == aterm_idx_current) {
                current_nr_timesteps++;
            } else {
                break;
            }
        }

        for (int i = tid; i < ALIGN(current_nr_timesteps, nr_threads); i += nr_threads) {
            int time = time_offset_local + i;

            float2 visXX[current_nr_channels];
            float2 visXY[current_nr_channels];
            float2 visYX[current_nr_channels];
            float2 visYY[current_nr_channels];

            for (int chan = 0; chan < current_nr_channels; chan++) {
                visXX[chan] = make_float2(0, 0);
                visXY[chan] = make_float2(0, 0);
                visYX[chan] = make_float2(0, 0);
                visYY[chan] = make_float2(0, 0);
            }

            float u, v, w;

            if (time < nr_timesteps) {
                u = uvw[time_offset_global + time].u;
                v = uvw[time_offset_global + time].v;
                w = uvw[time_offset_global + time].w;
            }

            __syncthreads();

            // Iterate pixels
            const int nr_pixels = subgrid_size * subgrid_size;
            int current_nr_pixels = BATCH_SIZE;
            for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
                current_nr_pixels = nr_pixels - pixel_offset < min(nr_threads, BATCH_SIZE) ?
                                    nr_pixels - pixel_offset : min(nr_threads, BATCH_SIZE);

                __syncthreads();

                // Prepare data
                for (int j = tid; j < current_nr_pixels; j += nr_threads) {
                    int y = (pixel_offset + j) / subgrid_size;
                    int x = (pixel_offset + j) % subgrid_size;

                    // Compute shifted position in subgrid
                    int x_src = (x + (subgrid_size/2)) % subgrid_size;
                    int y_src = (y + (subgrid_size/2)) % subgrid_size;

                    // Load spheroidal
                    float spheroidal_ = spheroidal[y * subgrid_size + x];

                    // Load pixels
                    float2 pixel[NR_POLARIZATIONS];
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned int pixel_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                        pixel[pol] = subgrid[pixel_idx] * spheroidal_;
                    }

                    // Apply aterm
                    int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_idx_current, station1, y, x);
                    int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_idx_current, station2, y, x);
                    float2 *aterm1 = (float2 *) &aterms[station1_idx];
                    float2 *aterm2 = (float2 *) &aterms[station2_idx];
                    apply_aterm_degridder(pixel, aterm1, aterm2);

                    // Store pixels in shared memory
                    shared[0][j] = *((float4 *) &pixel[0]);
                    shared[1][j] = *((float4 *) &pixel[2]);;

                    // Compute l,m,n and phase offset
                    const float l = compute_l(x, subgrid_size, image_size);
                    const float m = compute_m(y, subgrid_size, image_size);
                    const float n = compute_n(l, m);
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    // Store l,m,n and phase offset in shared memory
                    shared[2][j] = make_float4(l, m, n, phase_offset);
                } // end for j (pixels)

                __syncthreads();

                // Iterate current batch of pixels
                for (int k = 0; k < current_nr_pixels; k++) {
                    // Load pixels from shared memory
                    float2 apXX = make_float2(shared[0][k].x, shared[0][k].y);
                    float2 apXY = make_float2(shared[0][k].z, shared[0][k].w);
                    float2 apYX = make_float2(shared[1][k].x, shared[1][k].y);
                    float2 apYY = make_float2(shared[1][k].z, shared[1][k].w);

                    // Load l,m,n
                    float l = shared[2][k].x;
                    float m = shared[2][k].y;
                    float n = shared[2][k].z;

                    // Load phase offset
                    float phase_offset = shared[2][k].w;

                    // Compute phase index
                    float phase_index = u * l + v * m + w * n;

                    for (int chan = 0; chan < current_nr_channels; chan++) {
                        // Load wavenumber
                        float wavenumber = wavenumbers[channel_offset + chan];

                        // Compute phasor
                        float  phase  = (phase_index * wavenumber) - phase_offset;
                        float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                        // Multiply pixels by phasor
                        visXX[chan].x += phasor.x * apXX.x;
                        visXX[chan].y += phasor.x * apXX.y;
                        visXX[chan].x -= phasor.y * apXX.y;
                        visXX[chan].y += phasor.y * apXX.x;

                        visXY[chan].x += phasor.x * apXY.x;
                        visXY[chan].y += phasor.x * apXY.y;
                        visXY[chan].x -= phasor.y * apXY.y;
                        visXY[chan].y += phasor.y * apXY.x;

                        visYX[chan].x += phasor.x * apYX.x;
                        visYX[chan].y += phasor.x * apYX.y;
                        visYX[chan].x -= phasor.y * apYX.y;
                        visYX[chan].y += phasor.y * apYX.x;

                        visYY[chan].x += phasor.x * apYY.x;
                        visYY[chan].y += phasor.x * apYY.y;
                        visYY[chan].x -= phasor.y * apYY.y;
                        visYY[chan].y += phasor.y * apYY.x;
                    } // end for chan
                } // end for k (batch)
            } // end for pixel_offset

            for (int chan = 0; chan < current_nr_channels; chan++) {
                if (time < nr_timesteps) {
                    // Store visibility
                    const float scale = 1.0f / (subgrid_size * subgrid_size);
                    int idx_time = time_offset_global + time;
                    int idx_chan = channel_offset + chan;
                    int idx_vis = index_visibility(nr_channels, idx_time, idx_chan, 0);
                    float4 visA = make_float4(visXX[chan].x, visXX[chan].y, visXY[chan].x, visXY[chan].y);
                    float4 visB = make_float4(visYX[chan].x, visYX[chan].y, visYY[chan].x, visYY[chan].y);
                    float4 *vis_ptr = (float4 *) &visibilities[idx_vis];
                    vis_ptr[0] = visA * scale;
                    vis_ptr[1] = visB * scale;
                }
            } // end for chan
        } // end for time
    } // end for time_offset_local
} // end kernel_degridder_

#define KERNEL_DEGRIDDER(current_nr_channels) \
    for (; (channel_offset + current_nr_channels) <= nr_channels; channel_offset += current_nr_channels) { \
        kernel_degridder_<current_nr_channels>( \
            grid_size, subgrid_size, image_size, w_step, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, spheroidal, aterms, aterms_indices, metadata, subgrid); \
    }

#define GLOBAL_ARGUMENTS \
    const int                         grid_size,    \
    const int                         subgrid_size, \
    const float                       image_size,   \
    const float                       w_step,       \
    const int                         nr_channels,  \
    const int                         nr_stations,  \
    const UVW*           __restrict__ uvw,          \
    const float*         __restrict__ wavenumbers,  \
          float2*        __restrict__ visibilities, \
    const float*         __restrict__ spheroidal,   \
    const float2*        __restrict__ aterms,       \
    const int*           __restrict__ aterms_indices, \
    const Metadata*      __restrict__ metadata,     \
          float2*        __restrict__ subgrid

extern "C" {
__global__ void
    kernel_degridder_1(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(1)
}

__global__ void
    kernel_degridder_2(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(2)
}

__global__ void
    kernel_degridder_3(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(3)
}

__global__ void
    kernel_degridder_4(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(4)
}

__global__ void
    kernel_degridder_5(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(5)
}

__global__ void
    kernel_degridder_6(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(6)
}

__global__ void
    kernel_degridder_7(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(7)
}

__global__ void
    kernel_degridder_8(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(8)
}

__global__ void
    kernel_degridder_n(GLOBAL_ARGUMENTS)
{
    int channel_offset = 0;
    KERNEL_DEGRIDDER(8)
    KERNEL_DEGRIDDER(7)
    KERNEL_DEGRIDDER(6)
    KERNEL_DEGRIDDER(5)
    KERNEL_DEGRIDDER(4)
    KERNEL_DEGRIDDER(3)
    KERNEL_DEGRIDDER(2)
    KERNEL_DEGRIDDER(1)
}
} // end extern "C"
