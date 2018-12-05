#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

__shared__ float4 shared[3][BATCH_SIZE];

/*
    Kernel
*/
extern "C" {
__global__ void
__launch_bounds__(BLOCK_SIZE)
kernel_degridder_1(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         nr_channels,
    const int                         nr_stations,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
          float2*        __restrict__ visibilities,
    const float*         __restrict__ spheroidal,
    const float2*        __restrict__ aterm,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ subgrid)
{
    const unsigned UNROLL_TIME = 2;

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

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) m.coordinate.z + 0.5) * 2 * M_PI;

    // Iterate visibilities
    for (int time = tid; time < ALIGN(nr_timesteps, nr_threads); time += nr_threads * UNROLL_TIME) {
        float2 visXX[UNROLL_TIME];
        float2 visXY[UNROLL_TIME];
        float2 visYX[UNROLL_TIME];
        float2 visYY[UNROLL_TIME];

        for (unsigned i = 0; i < UNROLL_TIME; i++) {
            visXX[i] = make_float2(0, 0);
            visXY[i] = make_float2(0, 0);
            visYX[i] = make_float2(0, 0);
            visYY[i] = make_float2(0, 0);
        }

        float u[UNROLL_TIME];
        float v[UNROLL_TIME];
        float w[UNROLL_TIME];

        for (unsigned i = 0; i < UNROLL_TIME; i++) {
            unsigned time_ = time + i * nr_threads;

            if (time_ < nr_timesteps) {
                u[i] = uvw[time_offset_global + time_].u;
                v[i] = uvw[time_offset_global + time_].v;
                w[i] = uvw[time_offset_global + time_].w;
            }
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

                // Load pixels
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
                float2 pixelsXX = subgrid[idx_xx];
                float2 pixelsXY = subgrid[idx_xy];
                float2 pixelsYX = subgrid[idx_yx];
                float2 pixelsYY = subgrid[idx_yy];

                // Compute l,m,n and phase offset
                const float l = compute_l(x, subgrid_size, image_size);
                const float m = compute_m(y, subgrid_size, image_size);
                const float n = compute_n(l, m);
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Store values in shared memory
                shared[0][j] = make_float4(pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                shared[1][j] = make_float4(pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);
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

                // Load wavenumber
                float wavenumber = wavenumbers[0];

                // Iterate unrolled timesteps
                for (unsigned i = 0; i < UNROLL_TIME; i++) {
                    // Compute phase index
                    float phase_index = u[i] * l + v[i] * m + w[i] * n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                    // Multiply pixels by phasor
                    visXX[i].x += phasor.x * apXX.x;
                    visXX[i].y += phasor.x * apXX.y;
                    visXX[i].x -= phasor.y * apXX.y;
                    visXX[i].y += phasor.y * apXX.x;

                    visXY[i].x += phasor.x * apXY.x;
                    visXY[i].y += phasor.x * apXY.y;
                    visXY[i].x -= phasor.y * apXY.y;
                    visXY[i].y += phasor.y * apXY.x;

                    visYX[i].x += phasor.x * apYX.x;
                    visYX[i].y += phasor.x * apYX.y;
                    visYX[i].x -= phasor.y * apYX.y;
                    visYX[i].y += phasor.y * apYX.x;

                    visYY[i].x += phasor.x * apYY.x;
                    visYY[i].y += phasor.x * apYY.y;
                    visYY[i].x -= phasor.y * apYY.y;
                    visYY[i].y += phasor.y * apYY.x;
                }
            } // end for k (batch)
        } // end for j (pixels)

        for (unsigned i = 0; i < UNROLL_TIME; i++) {
            unsigned time_ = time + i * nr_threads;

            if (time_ < nr_timesteps) {
                // Store visibility
                const float scale = 1.0f / (nr_pixels);
                int idx_time = time_offset_global + time_;
                int idx_vis = index_visibility(1, idx_time, 0, 0);
                float4 visA = make_float4(visXX[i].x, visXX[i].y, visXY[i].x, visXY[i].y);
                float4 visB = make_float4(visYX[i].x, visYX[i].y, visYY[i].x, visYY[i].y);
                float4 *vis_ptr = (float4 *) &visibilities[idx_vis];
                vis_ptr[0] = visA * scale;
                vis_ptr[1] = visB * scale;
            }
        }
    } // end for time
} // end kernel_degridder_1
} // end extern "C"
