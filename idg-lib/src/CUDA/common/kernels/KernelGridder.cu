#include "math.cu"
#include "Types.h"

#include <assert.h>

#define BATCH_SIZE GRIDDER_BATCH_SIZE
#define BLOCK_SIZE GRIDDER_BLOCK_SIZE
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

#define MAX_NR_CHANNELS 8
#define UNROLL_PIXELS   2

__shared__ float4 shared[3][BATCH_SIZE];
__shared__ float wavenumbers_[MAX_NR_CHANNELS];

/*
    Kernel
*/
template<int current_nr_channels>
__device__ void
    kernel_gridder_(
    const int                           grid_size,
    const int                           subgrid_size,
    const float                         image_size,
    const float                         w_step,
    const int                           nr_channels,
    const int                           channel_offset,
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
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;
    int s = blockIdx.x;

	// Set subgrid to zero
	if (channel_offset == 0) {
		for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
			int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
			int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
			int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
			int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);
			subgrid[idx_xx] = make_float2(0, 0);
 			subgrid[idx_xy] = make_float2(0, 0);
 			subgrid[idx_yx] = make_float2(0, 0);
 			subgrid[idx_yy] = make_float2(0, 0);
		}
	}

    __syncthreads();

    // Load metadata for first subgrid
    const Metadata &m_0 = metadata[0];

	// Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int aterm_index = m.aterm_index;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) m.coordinate.z + 0.5) * 2 * M_PI;

	// Load wavenumbers
	for (int chan = tid; chan < current_nr_channels; chan += nr_threads) {
		wavenumbers_[chan] = wavenumbers[channel_offset + chan];
	}

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
        int current_nr_timesteps = BATCH_SIZE / MAX_NR_CHANNELS;
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
            current_nr_timesteps = nr_timesteps - time_offset_local < current_nr_timesteps ?
                                   nr_timesteps - time_offset_local : current_nr_timesteps;

            __syncthreads();

            // Load UVW
            for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
                UVW a = uvw[time_offset_global + time_offset_local + time];
                shared[2][time] = make_float4(a.u, a.v, a.w, 0);
            }

            // Load visibilities
            for (int i = tid; i < current_nr_timesteps*current_nr_channels; i += nr_threads) {
                int idx_time = time_offset_global + time_offset_local + (i / current_nr_channels);
                int idx_chan = channel_offset + (i % current_nr_channels);
                int idx_xx = index_visibility(nr_channels, idx_time, idx_chan, 0);
                int idx_xy = index_visibility(nr_channels, idx_time, idx_chan, 1);
                int idx_yx = index_visibility(nr_channels, idx_time, idx_chan, 2);
                int idx_yy = index_visibility(nr_channels, idx_time, idx_chan, 3);
                float2 a = visibilities[idx_xx];
                float2 b = visibilities[idx_xy];
                float2 c = visibilities[idx_yx];
                float2 d = visibilities[idx_yy];
                shared[0][i] = make_float4(a.x, a.y, b.x, b.y);
                shared[1][i] = make_float4(c.x, c.y, d.x, d.y);
            }

            __syncthreads();

            // Iterate current batch of timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                // Load UVW coordinates
                float u = shared[2][time].x;
                float v = shared[2][time].y;
                float w = shared[2][time].z;

                // Compute phase index and phase offset
                float phase_index[UNROLL_PIXELS];

                for (int j = 0; j < UNROLL_PIXELS; j++) {
                    phase_index[j]  = u*l[j] + v*m[j] + w*n[j];
                }

                #pragma unroll
                for (int chan = 0; chan < current_nr_channels; chan++) {
                    float wavenumber = wavenumbers_[chan];

                    // Load visibilities from shared memory
                    float4 a = shared[0][time*current_nr_channels+chan];
                    float4 b = shared[1][time*current_nr_channels+chan];
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
                } // end for chan
            } // end for time
        } // end for time_offset_local

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * nr_threads;
            if (i_ < subgrid_size * subgrid_size) {
                int y = i_ / subgrid_size;
                int x = i_ % subgrid_size;

                // Get aterm for station1
                int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x);
                float2 aXX1 = aterm[station1_idx + 0];
                float2 aXY1 = aterm[station1_idx + 1];
                float2 aYX1 = aterm[station1_idx + 2];
                float2 aYY1 = aterm[station1_idx + 3];

                // Get aterm for station2
                int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
                float2 aXX2 = aterm[station2_idx + 0];
                float2 aXY2 = aterm[station2_idx + 1];
                float2 aYX2 = aterm[station2_idx + 2];
                float2 aYY2 = aterm[station2_idx + 3];

                // Apply the conjugate transpose of the A-term
                apply_aterm(
                    conj(aXX1), conj(aYX1), conj(aXY1), conj(aYY1),
                    conj(aXX2), conj(aYX2), conj(aXY2), conj(aYY2),
                    uvXX[j], uvXY[j], uvYX[j], uvYY[j]);

                if (avg_aterm_correction) {
                    apply_avg_aterm_correction(
                        avg_aterm_correction + i*16,
                        uvXX[j], uvXY[j], uvYX[j], uvYY[j]);
                }

                // Load spheroidal
                float spheroidal_ = spheroidal[y * subgrid_size + x];

                // Compute shifted position in subgrid
                int x_dst = (x + (subgrid_size/2)) % subgrid_size;
                int y_dst = (y + (subgrid_size/2)) % subgrid_size;

                // Set subgrid value
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_dst, x_dst);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_dst, x_dst);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_dst, x_dst);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_dst, x_dst);
                subgrid[idx_xx] += uvXX[j] * spheroidal_;
                subgrid[idx_xy] += uvXY[j] * spheroidal_;
                subgrid[idx_yx] += uvYX[j] * spheroidal_;
                subgrid[idx_yy] += uvYY[j] * spheroidal_;
            }
        }
    } // end for i (pixels)
} // end kernel_gridder_

#define KERNEL_GRIDDER_TEMPLATE(current_nr_channels) \
    for (; (channel_offset + current_nr_channels) <= nr_channels; channel_offset += current_nr_channels) { \
        kernel_gridder_<current_nr_channels>( \
            grid_size, subgrid_size, image_size, w_step, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, spheroidal, aterm, avg_aterm_correction, metadata, subgrid); \
    }

extern "C" {

__global__ void
__launch_bounds__(BLOCK_SIZE)
    kernel_gridder(
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
    assert(subgrid_size * subgrid_size % UNROLL_PIXELS == 0);
	int channel_offset = 0;
	assert(MAX_NR_CHANNELS == 8);
	KERNEL_GRIDDER_TEMPLATE(8);
	KERNEL_GRIDDER_TEMPLATE(7);
	KERNEL_GRIDDER_TEMPLATE(6);
	KERNEL_GRIDDER_TEMPLATE(5);
	KERNEL_GRIDDER_TEMPLATE(4);
	KERNEL_GRIDDER_TEMPLATE(3);
	KERNEL_GRIDDER_TEMPLATE(2);
	KERNEL_GRIDDER_TEMPLATE(1);
}

} // end extern "C"
