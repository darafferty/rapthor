#include "math.cu"
#include "Types.h"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

__shared__ float4 visibilities_[BATCH_SIZE][2];
__shared__ float4 uvw_[BATCH_SIZE];

/*
    Kernel
*/
extern "C" {
__global__ void
__launch_bounds__(BLOCK_SIZE)
kernel_gridder_1(
    const int                           grid_size,
    const int                           subgrid_size,
    const float                         image_size,
    const float                         w_step,
    const int                           nr_channels,
    const int                           nr_stations,
    const UVW*             __restrict__ uvw,
    const float*           __restrict__ wavenumbers,
    const float2*          __restrict__ visibilities,
    const float*           __restrict__ spheroidal,
    const float2*          __restrict__ aterm,
    const float2*          __restrict__ avg_aterm_correction,
    const Metadata*        __restrict__ metadata,
          float2*          __restrict__ subgrid)
{
    const unsigned UNROLL_PIXELS = 4;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;
    int s = blockIdx.x;

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

    // Iterate all pixels in subgrid
    for (int i = tid; i < ALIGN(subgrid_size * subgrid_size, nr_threads); i += nr_threads * UNROLL_PIXELS) {
        // Private pixels
        float2 uvXX[UNROLL_PIXELS];
        float2 uvXY[UNROLL_PIXELS];
        float2 uvYX[UNROLL_PIXELS];
        float2 uvYY[UNROLL_PIXELS];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            uvXX[j] = make_float2(0, 0);
            uvXY[j] = make_float2(0, 0);
            uvYX[j] = make_float2(0, 0);
            uvYY[j] = make_float2(0, 0);
        }

        // Compute l,m,n, phase_offset
        float l[UNROLL_PIXELS];
        float m[UNROLL_PIXELS];
        float n[UNROLL_PIXELS];
        float phase_offset[UNROLL_PIXELS];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * nr_threads;
            int y = i_ / subgrid_size;
            int x = i_ % subgrid_size;
            l[j] = compute_l(x, subgrid_size, image_size);
            m[j] = compute_m(y, subgrid_size, image_size);
            n[j] = compute_n(l[j], m[j]);
            phase_offset[j] = u_offset*l[j] + v_offset*m[j] + w_offset*n[j];
        }

        // Iterate timesteps
        int current_nr_timesteps = BATCH_SIZE;
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
            current_nr_timesteps = nr_timesteps - time_offset_local < current_nr_timesteps ?
                                   nr_timesteps - time_offset_local : current_nr_timesteps;

            __syncthreads();

            // Load UVW
            for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
                UVW a = uvw[time_offset_global + time_offset_local + time];
                uvw_[time] = make_float4(a.u, a.v, a.w, 0);
            }

            // Load visibilities
            for (int i = tid; i < current_nr_timesteps*2; i += nr_threads) {
                int j = i % 2; // one thread loads either upper or lower float4 part of visibility
                int k = i / 2;
                int idx_time = time_offset_global + time_offset_local + k;
                int idx_vis = index_visibility(1, idx_time, 0, 0);
                float4 *vis_ptr = (float4 *) &visibilities[idx_vis];
                visibilities_[k][j] = vis_ptr[j];
            }

            __syncthreads();

            // Iterate current batch of timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                // Load UVW coordinates
                float u = uvw_[time].x;
                float v = uvw_[time].y;
                float w = uvw_[time].z;

                // Compute phase index
                float phase_index[UNROLL_PIXELS];

                for (int j = 0; j < UNROLL_PIXELS; j++) {
                    phase_index[j]  = u*l[j] + v*m[j] + w*n[j];
                }

                // Load wavenumber
                float wavenumber = wavenumbers[0];

                // Load visibilities from shared memory
                float4 a = visibilities_[time][0];
                float4 b = visibilities_[time][1];
                float2 visXX = make_float2(a.x, a.y);
                float2 visXY = make_float2(a.z, a.w);
                float2 visYX = make_float2(b.x, b.y);
                float2 visYY = make_float2(b.z, b.w);

                for (int j = 0; j < UNROLL_PIXELS; j++) {
                    // Compute phasor
                    float phase = phase_offset[j] - (phase_index[j] * wavenumber);
                    float2 phasor = make_float2(cosf(phase), sinf(phase));

                    // Multiply visibility by phasor
                    uvXX[j].x += phasor.x * visXX.x;
                    uvXX[j].y += phasor.x * visXX.y;
                    uvXX[j].x -= phasor.y * visXX.y;
                    uvXX[j].y += phasor.y * visXX.x;

                    uvXY[j].x += phasor.x * visXY.x;
                    uvXY[j].y += phasor.x * visXY.y;
                    uvXY[j].x -= phasor.y * visXY.y;
                    uvXY[j].y += phasor.y * visXY.x;

                    uvYX[j].x += phasor.x * visYX.x;
                    uvYX[j].y += phasor.x * visYX.y;
                    uvYX[j].x -= phasor.y * visYX.y;
                    uvYX[j].y += phasor.y * visYX.x;

                    uvYY[j].x += phasor.x * visYY.x;
                    uvYY[j].y += phasor.x * visYY.y;
                    uvYY[j].x -= phasor.y * visYY.y;
                    uvYY[j].y += phasor.y * visYY.x;
                }
            } // end for time
        } // end for time_offset_local

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * nr_threads;
            if (i_ < subgrid_size * subgrid_size) {
                int y = i_ / subgrid_size;
                int x = i_ % subgrid_size;

                // Compute shifted position in subgrid
                int x_dst = (x + (subgrid_size/2)) % subgrid_size;
                int y_dst = (y + (subgrid_size/2)) % subgrid_size;

                // Set subgrid value
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_dst, x_dst);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_dst, x_dst);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_dst, x_dst);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_dst, x_dst);
                subgrid[idx_xx] = uvXX[j];;
                subgrid[idx_xy] = uvXY[j];;
                subgrid[idx_yx] = uvYX[j];;
                subgrid[idx_yy] = uvYY[j];;
            }
        }
    } // end for i (pixels)
} // end kernel_gridder_1
} // end extern "C"
