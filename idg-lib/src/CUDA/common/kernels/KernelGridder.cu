// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "math.cu"
#include "Types.h"
#include "KernelGridder.cuh"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))


/**
 * Tuning parameters:
 *  - BLOCK_SIZE_X
 *  - NUM_BLOCKS
 *  - UNROLL_PIXELS
 *
 * This kernel is tuned for these
 * architectures (__CUDA_ARCH__):
 *  - Ampere (800, 860)
 *  - Turing (750)
 *  - Volta (700)
 *  - Pascal (610)
 *  - Maxwell (520)
 *  - Kepler (350)
 *
 *  Derived parameters:
 *  - NUM_THREADS = BLOCK_SIZE_X
 *    -> the thread block is 1D
 *  - BATCH_SIZE = NUM_THREADS
 *    -> setting BATCH_SIZE as a multiple of
 *       NUM_THREADS does not improve performance
**/
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X KernelGridder::block_size_x
#endif
#define NUM_THREADS BLOCK_SIZE_X

#ifndef NUM_BLOCKS
#if __CUDA_ARCH__ >= 800
#define NUM_BLOCKS 2
#else
#define NUM_BLOCKS 4
#endif
#endif

#ifndef UNROLL_PIXELS
#define UNROLL_PIXELS 4
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE NUM_THREADS
#endif


/*
    Shared memory
*/
__shared__ float4 visibilities_[2][BATCH_SIZE];
__shared__ float4 uvw_[BATCH_SIZE];


/*
    Kernels
*/
template<unsigned nr_polarizations>
__device__ void update_pixel(
    const unsigned             subgrid_size,
    const unsigned             nr_stations,
    const unsigned             s,
    const unsigned             y,
    const unsigned             x,
    const unsigned             aterm_index,
    const unsigned             station1,
    const unsigned             station2,
    const float2* __restrict__ aterms,
          float2* __restrict__ pixel,
          float2* __restrict__ pixel_sum)
{
    // Apply aterm
    int station1_idx = index_aterm(subgrid_size, 4, nr_stations, aterm_index, station1, y, x, 0);
    int station2_idx = index_aterm(subgrid_size, 4, nr_stations, aterm_index, station2, y, x, 0);
    float2 *aterm1 = (float2 *) &aterms[station1_idx];
    float2 *aterm2 = (float2 *) &aterms[station2_idx];
    apply_aterm_gridder(pixel, aterm1, aterm2);

    // Update pixel
    if (nr_polarizations == 4) {
        // Full Stokes
        for (unsigned pol = 0; pol < nr_polarizations; pol++) {
            pixel_sum[pol] += pixel[pol];
        }
    } else if (nr_polarizations == 1) {
        // Stokes-I only
        pixel_sum[0] += pixel[0];
        pixel_sum[3] += pixel[3];
    }
}

template<unsigned nr_polarizations>
__device__ void update_subgrid(
    const unsigned             subgrid_size,
    const unsigned             s,
    const unsigned             y,
    const unsigned             x,
    const float*  __restrict__ taper,
    const float2* __restrict__ avg_aterm,
          float2* __restrict__ pixel,
          float2* __restrict__ subgrid)
{
    // Compute shifted position in subgrid
    int x_dst = (x + (subgrid_size/2)) % subgrid_size;
    int y_dst = (y + (subgrid_size/2)) % subgrid_size;

    // Pixel index
    int i = y * subgrid_size + x;

    // Apply average aterm correction
    if (avg_aterm) {
        apply_avg_aterm_correction_(
            avg_aterm + i*16, pixel);
    }

    // Load taper
    const float taper_ = taper[i];

    // Update subgrid
    if (nr_polarizations == 4) {
        for (int pol = 0; pol < 4; pol++) {
            int idx = index_subgrid(4, subgrid_size, s, pol, y_dst, x_dst);
            subgrid[idx] = pixel[pol] * taper_;
        }
    } else if (nr_polarizations == 1) {
        int idx = index_subgrid(1, subgrid_size, s, 0, y_dst, x_dst);
        subgrid[idx] = (pixel[0] + pixel[3]) * taper_ * 0.5f;
    }
}

template<int nr_polarizations>
__device__ void
    kernel_gridder_(
    const int                           time_offset,
    const int                           grid_size,
    const int                           subgrid_size,
    const float                         image_size,
    const float                         w_step,
    const float                         shift_l,
    const float                         shift_m,
    const int                           nr_channels,
    const int                           current_nr_channels,
    const int                           channel_offset,
    const int                           nr_stations,
    const UVW<float>*      __restrict__ uvw,
    const float*           __restrict__ wavenumbers,
    const float2*          __restrict__ visibilities,
    const float*           __restrict__ taper,
    const float2*          __restrict__ aterms,
    const unsigned int*    __restrict__ aterm_indices,
    const Metadata*        __restrict__ metadata,
    const float2*          __restrict__ avg_aterm,
          float2*          __restrict__ subgrid)
{
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int s = blockIdx.x;

	// Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int time_offset_global = m.time_index - time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) m.coordinate.z + 0.5) * 2 * M_PI;

    // Iterate all pixels in subgrid
    for (int i = tid; i < ALIGN(subgrid_size * subgrid_size, NUM_THREADS); i += num_threads * UNROLL_PIXELS) {
        float2 pixel_cur[UNROLL_PIXELS][4];
        float2 pixel_sum[UNROLL_PIXELS][4];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            for (int k = 0; k < 4; k++) {
                pixel_cur[j][k] = make_float2(0, 0);
                pixel_sum[j][k] = make_float2(0, 0);
            }
        }

        // Initialize aterm index to first timestep
        unsigned int aterm_idx_previous = aterm_indices[time_offset_global];

        // Compute l,m,n, phase_offset
        float l_index[UNROLL_PIXELS];
        float m_index[UNROLL_PIXELS];
        float n[UNROLL_PIXELS];
        float phase_offset[UNROLL_PIXELS];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * num_threads;
            int y = i_ / subgrid_size;
            int x = i_ % subgrid_size;
            float l_offset = compute_l(x, subgrid_size, image_size);
            float m_offset = compute_m(y, subgrid_size, image_size);
            l_index[j] = l_offset + shift_l;
            m_index[j] = m_offset - shift_m;
            n[j] = compute_n(l_index[j], m_index[j]);
            phase_offset[j] = u_offset*l_offset + v_offset*m_offset + w_offset*n[j];
        }

        // Iterate timesteps
        int current_nr_timesteps = max(1, BATCH_SIZE / current_nr_channels);
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
            current_nr_timesteps = min(current_nr_timesteps, nr_timesteps - time_offset_local);

            __syncthreads();

            // Load UVW
            for (int j = tid; j < current_nr_timesteps; j += num_threads) {
                int time = time_offset_local + j;
                if (time < nr_timesteps) {
                    int idx_time = time_offset_global + time;
                    UVW<float> a = uvw[idx_time];
                    uvw_[j] = make_float4(a.u, a.v, a.w, 0);
                }
            }

            // Load visibilities
            if (nr_polarizations == 4) {
                for (int v = tid; v < current_nr_timesteps*current_nr_channels*2; v += num_threads) {
                    int j = v % 2; // one thread loads either upper or lower float4 part of visibility
                    int k = v / 2;
                    int time = time_offset_local + (k / current_nr_channels);
                    int idx_time = time_offset_global + time;
                    int idx_chan = channel_offset + (k % current_nr_channels);
                    long idx_vis = index_visibility(4, nr_channels, idx_time, idx_chan, 0);
                    if (time < nr_timesteps) {
                        float4 *vis_ptr = (float4 *) &visibilities[idx_vis];
                        visibilities_[j][k] = vis_ptr[j];
                    }
                }
            } else if (nr_polarizations == 1) {
                // Use only visibilities_[0][*]
                for (int k = tid; k < current_nr_timesteps*current_nr_channels; k += num_threads) {
                    int time = time_offset_local + (k / current_nr_channels);
                    int idx_time = time_offset_global + time;
                    int idx_chan = channel_offset + (k % current_nr_channels);
                    long idx_vis = index_visibility(2, nr_channels, idx_time, idx_chan, 0);
                    if (time < nr_timesteps) {
                        float4 *vis_ptr = (float4 *) &visibilities[idx_vis];
                        visibilities_[0][k].x = vis_ptr[0].x;
                        visibilities_[0][k].y = vis_ptr[0].y;
                        visibilities_[0][k].z = 0.0;
                        visibilities_[0][k].w = 0.0;
                        visibilities_[1][k].x = 0.0;
                        visibilities_[1][k].y = 0.0;
                        visibilities_[1][k].z = vis_ptr[0].z;
                        visibilities_[1][k].w = vis_ptr[0].w;
                    }
                }
            }

            __syncthreads();

            // Iterate current batch of timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {

                // Get aterm index for current timestep
                int time_current = time_offset_global + time_offset_local + time;
                const unsigned int aterm_idx_current = aterm_indices[time_current];

                // Determine whether aterm has changed
                bool aterm_changed = aterm_idx_previous != aterm_idx_current;

                if (aterm_changed) {
                    for (int j = 0; j < UNROLL_PIXELS; j++) {
                        int i_ = i + j * num_threads;
                        int y = i_ / subgrid_size;
                        int x = i_ % subgrid_size;

                        // Update subgrid
                        if (y < subgrid_size) {
                            update_pixel<nr_polarizations>(
                                subgrid_size, nr_stations, s, y, x,
                                aterm_idx_previous, station1, station2, aterms,
                                &pixel_cur[j][0], &pixel_sum[j][0]);
                        }

                        // Reset pixel
                        for (int pol = 0; pol < 4; pol++) {
                            pixel_cur[j][pol] = make_float2(0, 0);
                        }
                    }

                    // Update aterm index
                    aterm_idx_previous = aterm_idx_current;
                }

                // Load UVW coordinates
                float u = uvw_[time].x;
                float v = uvw_[time].y;
                float w = uvw_[time].z;

                // Compute phase index
                float phase_index[UNROLL_PIXELS];

                for (int j = 0; j < UNROLL_PIXELS; j++) {
                    phase_index[j] = -(u*l_index[j] + v*m_index[j] + w*n[j]);
                }

                // Compute pixel
                #if USE_EXTRAPOLATE
                compute_reduction_extrapolate<UNROLL_PIXELS>(
                    current_nr_channels, nr_polarizations,
                    wavenumbers + channel_offset, phase_index, phase_offset,
                    &visibilities_[0][time*current_nr_channels],
                    &visibilities_[1][time*current_nr_channels],
                    reinterpret_cast<float2*>(pixel_cur), 1, 1);
                #else
                compute_reduction(
                    current_nr_channels, UNROLL_PIXELS, nr_polarizations,
                    wavenumbers + channel_offset, phase_index, phase_offset,
                    &visibilities_[0][time*current_nr_channels],
                    &visibilities_[1][time*current_nr_channels],
                    reinterpret_cast<float2*>(pixel_cur), 1, 1);
                #endif
            } // end for time
        } // end for time_offset_local

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * num_threads;
            int y = i_ / subgrid_size;
            int x = i_ % subgrid_size;

            if (y < subgrid_size) {
                update_pixel<nr_polarizations>(
                    subgrid_size, nr_stations, s, y, x,
                    aterm_idx_previous, station1, station2, aterms,
                    &pixel_cur[j][0], &pixel_sum[j][0]);
                update_subgrid<nr_polarizations>(
                    subgrid_size, s, y, x,
                    taper, avg_aterm, &pixel_sum[j][0], subgrid);
            }
        }
    } // end for i (pixels)
} // end kernel_gridder_

extern "C" {
__global__
#if NUM_BLOCKS > 0
__launch_bounds__(NUM_THREADS, NUM_BLOCKS)
#endif
void kernel_gridder(
        const int                        time_offset,
        const int                        nr_polarizations,
        const int                        grid_size,
        const int                        subgrid_size,
        const float                      image_size,
        const float                      w_step,
        const float                      shift_l,
        const float                      shift_m,
        const int                        nr_channels,
        const int                        nr_stations,
        const UVW<float>*   __restrict__ uvw,
        const float*        __restrict__ wavenumbers,
        const float2*       __restrict__ visibilities,
        const float*        __restrict__ taper,
        const float2*       __restrict__ aterms,
        const unsigned int* __restrict__ aterm_indices,
        const Metadata*     __restrict__ metadata,
        const float2*       __restrict__ avg_aterm,
              float2*       __restrict__ subgrid)
{
    int s                   = blockIdx.x;
    const Metadata &m       = metadata[s];
    const int channel_begin = m.channel_begin;
    const int channel_end   = m.channel_end;

    const int channel_offset      = channel_begin;
    const int current_nr_channels = channel_end - channel_begin;

    if (nr_polarizations == 1) {
        kernel_gridder_<1>(
            time_offset, grid_size, subgrid_size, image_size, w_step,
            shift_l, shift_m, nr_channels, current_nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, taper, aterms, aterm_indices,
            metadata, avg_aterm, subgrid);
    }  else {
        kernel_gridder_<4>(
            time_offset, grid_size, subgrid_size, image_size, w_step,
            shift_l, shift_m, nr_channels, current_nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, taper, aterms, aterm_indices,
            metadata, avg_aterm, subgrid);
    }
}
} // end extern "C"
