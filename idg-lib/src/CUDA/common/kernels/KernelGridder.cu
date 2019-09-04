#include "math.cu"
#include "Types.h"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

#define MAX_NR_CHANNELS 8
#define UNROLL_PIXELS   4

__shared__ float4 visibilities_[BATCH_SIZE][2];
__shared__ float4 uvw_[BATCH_SIZE];
__shared__ float wavenumbers_[MAX_NR_CHANNELS];

/*
    Kernel
*/
__device__ void update_subgrid(
    const unsigned             subgrid_size,
    const unsigned             nr_stations,
    const unsigned             tid,
    const unsigned             nr_threads,
    const unsigned             s,
    const unsigned             y,
    const unsigned             x,
    const unsigned             aterm_index,
    const unsigned             station1,
    const unsigned             station2,
    float2                     pixelXX,
    float2                     pixelXY,
    float2                     pixelYX,
    float2                     pixelYY,
    const float2* __restrict__ aterms,
          float2* __restrict__ subgrid)
{
    float2 pixel[4] = {pixelXX, pixelXY, pixelYX, pixelYY};

    // Compute shifted position in subgrid
    int x_dst = (x + (subgrid_size/2)) % subgrid_size;
    int y_dst = (y + (subgrid_size/2)) % subgrid_size;

    // Apply aterm
    int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, 0);
    int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, 0);
    float2 *aterm1 = (float2 *) &aterms[station1_idx];
    float2 *aterm2 = (float2 *) &aterms[station2_idx];
    apply_aterm_gridder(pixel, aterm1, aterm2);

    // Update subgrid
    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
        int idx = index_subgrid(subgrid_size, s, pol, y_dst, x_dst);
        subgrid[idx] += pixel[pol];
    }
}

__device__ void finalize_subgrid(
    const unsigned                      subgrid_size,
    const float*           __restrict__ spheroidal,
    const float2*          __restrict__ avg_aterm_correction,
    const Metadata*        __restrict__ metadata,
          float2*          __restrict__ subgrid)

{
    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s = blockIdx.x;

    __syncthreads();

    // Apply average aterm correction and spheroidal
    for (int i = tid; i < (subgrid_size * subgrid_size); i += nr_threads) {
        unsigned y = i / subgrid_size;
        unsigned x = i % subgrid_size;

        if (y < subgrid_size) {
            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Compute pixel indices
            int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
            int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
            int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
            int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);

            // Load pixels
            float2 pixelXX = subgrid[idx_xx];
            float2 pixelXY = subgrid[idx_xy];
            float2 pixelYX = subgrid[idx_yx];
            float2 pixelYY = subgrid[idx_yy];

            // Apply average aterm correction
            if (avg_aterm_correction) {
                apply_avg_aterm_correction(
                    avg_aterm_correction + i*16,
                    pixelXX, pixelXY, pixelYX, pixelYY);
            }

            // Load spheroidal
            float spheroidal_ = spheroidal[i];

            // Update subgrid
            subgrid[idx_xx] = pixelXX * spheroidal_;
            subgrid[idx_xy] = pixelXY * spheroidal_;
            subgrid[idx_yx] = pixelYX * spheroidal_;
            subgrid[idx_yy] = pixelYY * spheroidal_;
        }
    } // end for i (pixels)
}

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
    const UVW<float>*      __restrict__ uvw,
    const float*           __restrict__ wavenumbers,
    const float2*          __restrict__ visibilities,
    const float2*          __restrict__ aterms,
    const int*             __restrict__ aterms_indices,
    const Metadata*        __restrict__ metadata,
          float2*          __restrict__ subgrid)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Load metadata for first subgrid
    const Metadata &m0 = metadata[0];

	// Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int time_offset_global = m.time_index - m0.time_index;
    const int nr_timesteps = m.nr_timesteps;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int channel_begin = m.channel_begin; \

	// Set subgrid to zero
	if (channel_offset == channel_begin) {
		for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;
            if (y < subgrid_size) {
			    int idx_xx = index_subgrid(subgrid_size, s, 0, y, x);
			    int idx_xy = index_subgrid(subgrid_size, s, 1, y, x);
			    int idx_yx = index_subgrid(subgrid_size, s, 2, y, x);
			    int idx_yy = index_subgrid(subgrid_size, s, 3, y, x);
                subgrid[idx_xx] = make_float2(0, 0);
                subgrid[idx_xy] = make_float2(0, 0);
                subgrid[idx_yx] = make_float2(0, 0);
                subgrid[idx_yy] = make_float2(0, 0);
            }
		}
	}

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
        float2 pixelsXX[UNROLL_PIXELS];
        float2 pixelsXY[UNROLL_PIXELS];
        float2 pixelsYX[UNROLL_PIXELS];
        float2 pixelsYY[UNROLL_PIXELS];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            pixelsXX[j] = make_float2(0, 0);
            pixelsXY[j] = make_float2(0, 0);
            pixelsYX[j] = make_float2(0, 0);
            pixelsYY[j] = make_float2(0, 0);
        }

        // Initialize aterm index to first timestep
        int aterm_idx_previous = aterms_indices[time_offset_global];

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
        int current_nr_timesteps = BATCH_SIZE / current_nr_channels;
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
            current_nr_timesteps = nr_timesteps - time_offset_local < current_nr_timesteps ?
                                   nr_timesteps - time_offset_local : current_nr_timesteps;

            __syncthreads();

            // Load UVW
            for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
                UVW<float> a = uvw[time_offset_global + time_offset_local + time];
                uvw_[time] = make_float4(a.u, a.v, a.w, 0);
            }

            // Load visibilities
            for (int v = tid; v < current_nr_timesteps*current_nr_channels*2; v += nr_threads) {
                int j = v % 2; // one thread loads either upper or lower float4 part of visibility
                int k = v / 2;
                int idx_time = time_offset_global + time_offset_local + (k / current_nr_channels);
                int idx_chan = channel_offset + (k % current_nr_channels);
                int idx_vis = index_visibility(nr_channels, idx_time, idx_chan, 0);
                float4 *vis_ptr = (float4 *) &visibilities[idx_vis];
                visibilities_[k][j] = vis_ptr[j];
            }

            __syncthreads();

            // Iterate current batch of timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {

                // Get aterm index for current timestep
                int time_current = time_offset_global + time_offset_local + time;
                int aterm_idx_current = aterms_indices[time_current];

                // Determine whether aterm has changed
                bool aterm_changed = aterm_idx_previous != aterm_idx_current;

                if (aterm_changed) {
                    for (int j = 0; j < UNROLL_PIXELS; j++) {
                        int i_ = i + j * nr_threads;
                        int y = i_ / subgrid_size;
                        int x = i_ % subgrid_size;

                        // Update subgrid
                        if (y < subgrid_size) {
                            update_subgrid(
                                subgrid_size, nr_stations, tid, nr_threads, s, y, x,
                                aterm_idx_previous, station1, station2,
                                pixelsXX[j], pixelsXY[j], pixelsYX[j], pixelsYY[j],
                                aterms, subgrid);
                        }

                        // Reset pixel
                        pixelsXX[j] = make_float2(0, 0);
                        pixelsXY[j] = make_float2(0, 0);
                        pixelsYX[j] = make_float2(0, 0);
                        pixelsYY[j] = make_float2(0, 0);
                    }

                    // Update aterm index
                    aterm_idx_previous = aterm_idx_current;
                }

                // Load UVW coordinates
                float u = uvw_[time].x;
                float v = uvw_[time].y;
                float w = uvw_[time].z;

                // Compute phase index and phase offset
                float phase_index[UNROLL_PIXELS];

                for (int j = 0; j < UNROLL_PIXELS; j++) {
                    phase_index[j]  = u*l[j] + v*m[j] + w*n[j];
                }

                #pragma unroll
                for (int chan = 0; chan < current_nr_channels; chan++) {
                    float wavenumber = wavenumbers_[chan];

                    // Load visibilities from shared memory
                    float4 a = visibilities_[time*current_nr_channels+chan][0];
                    float4 b = visibilities_[time*current_nr_channels+chan][1];
                    float2 visXX = make_float2(a.x, a.y);
                    float2 visXY = make_float2(a.z, a.w);
                    float2 visYX = make_float2(b.x, b.y);
                    float2 visYY = make_float2(b.z, b.w);

                    for (int j = 0; j < UNROLL_PIXELS; j++) {
                        // Compute phasor
                        float phase = phase_offset[j] - (phase_index[j] * wavenumber);
                        #if defined(RAW_SINCOS)
                        float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));
                        #else
                        float2 phasor = make_float2(cosf(phase), sinf(phase));
                        #endif

                        // Multiply visibility by phasor
                        cmac(pixelsXX[j], phasor, visXX);
                        cmac(pixelsXY[j], phasor, visXY);
                        cmac(pixelsYX[j], phasor, visYX);
                        cmac(pixelsYY[j], phasor, visYY);
                    }
                } // end for chan
            } // end for time
        } // end for time_offset_local

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * nr_threads;
            int y = i_ / subgrid_size;
            int x = i_ % subgrid_size;

            if (y < subgrid_size) {
                update_subgrid(
                    subgrid_size, nr_stations, tid, nr_threads, s, y, x,
                    aterm_idx_previous, station1, station2,
                    pixelsXX[j], pixelsXY[j], pixelsYX[j], pixelsYY[j],
                    aterms, subgrid);
            }
        }
    } // end for i (pixels)

    __syncthreads();
} // end kernel_gridder_

#define LOAD_METADATA \
    int s                   = blockIdx.x; \
    const Metadata &m       = metadata[s]; \
    const int channel_begin = m.channel_begin; \
    const int channel_end   = m.channel_end;

#define KERNEL_GRIDDER(current_nr_channels) \
    for (; (channel_offset + current_nr_channels) <= channel_end; channel_offset += current_nr_channels) { \
        kernel_gridder_<current_nr_channels>( \
            grid_size, subgrid_size, image_size, w_step, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, aterms, aterms_indices, metadata, subgrid); \
    }

#define FINALIZE_SUBGRID \
    finalize_subgrid( \
    subgrid_size, spheroidal, avg_aterm_correction, metadata, subgrid);


#define GLOBAL_ARGUMENTS \
    const int                         grid_size,    \
    const int                         subgrid_size, \
    const float                       image_size,   \
    const float                       w_step,       \
    const int                         nr_channels,  \
    const int                         nr_stations,  \
    const UVW<float>*    __restrict__ uvw,          \
    const float*         __restrict__ wavenumbers,  \
          float2*        __restrict__ visibilities, \
    const float*         __restrict__ spheroidal,   \
    const float2*        __restrict__ aterms,       \
    const int*           __restrict__ aterms_indices,       \
    const float2*        __restrict__ avg_aterm_correction, \
    const Metadata*      __restrict__ metadata,             \
          float2*        __restrict__ subgrid

extern "C" {
__global__ void
    kernel_gridder(GLOBAL_ARGUMENTS)
{
    LOAD_METADATA
    int channel_offset = channel_begin;
    KERNEL_GRIDDER(8)
    KERNEL_GRIDDER(7)
    KERNEL_GRIDDER(6)
    KERNEL_GRIDDER(5)
    KERNEL_GRIDDER(4)
    KERNEL_GRIDDER(3)
    KERNEL_GRIDDER(2)
    KERNEL_GRIDDER(1)
    FINALIZE_SUBGRID
}
} // end extern "C"
