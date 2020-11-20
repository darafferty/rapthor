// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS      8
#define BATCH_SIZE_PIXELS 64

inline __device__ long index_sums(
    unsigned int total_nr_timesteps, // number of timesteps for all baselines
    unsigned int nr_channels,        // number channels for a single baseline
    unsigned int term_nr,
    unsigned int pol,
    unsigned int time,
    unsigned int chan)
{
    // sums: [MAX_NR_TERMS][NR_POLARIZATIONS][TOTAL_NR_TIMESTEPS][NR_CHANNELS]
    return term_nr * NR_POLARIZATIONS * total_nr_timesteps * nr_channels +
           pol * total_nr_timesteps * nr_channels +
           time * nr_channels +
           chan;
}

inline __device__ long index_lmnp(
        unsigned int subgrid_size,
        unsigned int s,
        unsigned int y,
        unsigned int x)
{
    // lmnp: [NR_SUBGRIDS][SUBGRIDSIZE][SUBGRIDSIZE]
    return s * subgrid_size * subgrid_size +
           y * subgrid_size + x;
}

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


__global__ void kernel_calibrate_sums(
    const int                      subgrid_size,
    const float                    image_size,
    const int                      total_nr_timesteps,
    const int                      nr_channels,
    const int                      nr_stations,
    const int                      term_offset,
    const int                      current_nr_terms,
    const int                      nr_terms,
    const UVW<float>* __restrict__ uvw,
    const float*      __restrict__ wavenumbers,
    const float2*     __restrict__ aterm,
    const float2*     __restrict__ aterm_derivatives,
    const int*        __restrict__ aterm_indices,
    const Metadata*   __restrict__ metadata,
    const float2*     __restrict__ subgrid,
          float2*     __restrict__ sums,
    const float4*     __restrict__ lmnp)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Load metadata for first subgrid
    const Metadata &m0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset_global = m.time_index - m0.time_index;
    const unsigned int station2           = m.baseline.station2;
    const unsigned int nr_timesteps       = m.nr_timesteps;
    const unsigned int channel_begin      = m.channel_begin;
    const unsigned int channel_end        = m.channel_end;
    const unsigned int nr_channels_local  = channel_end - channel_begin;

    // Shared memory
    __shared__ float4 pixels_[MAX_NR_TERMS][BATCH_SIZE_PIXELS][2];
    __shared__ float4 lmnp_[BATCH_SIZE_PIXELS];

    // Iterate timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        int aterm_idx = aterm_indices[time_offset_global + time_offset_local];

        // Determine number of timesteps to process
        current_nr_timesteps = 0;
        for (int time = time_offset_local; time < nr_timesteps; time++) {
            if (aterm_indices[time_offset_global + time] == aterm_idx) {
                current_nr_timesteps++;
            } else {
                break;
            }
        }

        // Iterate batch of visibilities from the same timeslot
        for (int i = tid; i < ALIGN(current_nr_timesteps * nr_channels_local, nr_threads); i += nr_threads) {
            unsigned int time_idx_batch  = (i / nr_channels_local);
            unsigned int chan_idx_local  = (i % nr_channels_local) + channel_begin;
            unsigned int time_idx_local  = time_offset_local + time_idx_batch;
            unsigned int time_idx_global = time_offset_global + time_idx_local;

            // Load UVW
            float u, v, w;
            if (time_idx_batch < current_nr_timesteps) {
                u = uvw[time_idx_global].u;
                v = uvw[time_idx_global].v;
                w = uvw[time_idx_global].w;
            }

            // Load wavenumber
            float wavenumber = wavenumbers[chan_idx_local];


            // Accumulate sums in registers
            float2 sum[MAX_NR_TERMS][NR_POLARIZATIONS] = {0, 0};

            // Iterate all pixels
            for (unsigned int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += BATCH_SIZE_PIXELS) {

                __syncthreads();

                for (unsigned int j = tid; j < BATCH_SIZE_PIXELS; j += nr_threads) {
                    unsigned int y = (pixel_offset + j) / subgrid_size;
                    unsigned int x = (pixel_offset + j) % subgrid_size;

                    for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
                        // Reset pixel to zero
                        pixels_[term_nr][j][0] = make_float4(0, 0, 0, 0);
                        pixels_[term_nr][j][1] = make_float4(0, 0, 0, 0);

                        // Prepare batch
                        if (y < subgrid_size) {
                            // Compute shifted position in subgrid
                            unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                            unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                             // Load pixel and aterms
                            float2 pixel[NR_POLARIZATIONS];
                            float2 aterm1[NR_POLARIZATIONS];
                            float2 aterm2[NR_POLARIZATIONS];
                            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                unsigned int term_idx   = term_offset + term_nr;
                                unsigned int pixel_idx  = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                                unsigned int aterm1_idx = index_aterm_transposed(subgrid_size, nr_terms, aterm_idx, term_idx, y, x, pol);
                                unsigned int aterm2_idx = index_aterm_transposed(subgrid_size, nr_stations, aterm_idx, station2, y, x, pol);
                                pixel[pol]  = subgrid[pixel_idx];
                                aterm1[pol] = aterm_derivatives[aterm1_idx];
                                aterm2[pol] = aterm[aterm2_idx];
                            }

                            // Apply aterm
                            apply_aterm_degridder(pixel, aterm1, aterm2);

                            // Store pixel in shared memory
                            pixels_[term_nr][j][0] = make_float4(pixel[0].x, pixel[0].y, pixel[1].x, pixel[1].y);
                            pixels_[term_nr][j][1] = make_float4(pixel[2].x, pixel[2].y, pixel[3].x, pixel[3].y);
                        } // end if
                    } // end for term_nr

                    // Load l,m,n and phase_offset into shared memory
                    if (y < subgrid_size) {
                        unsigned int lmnp_idx = index_lmnp(subgrid_size, s, y, x);
                        lmnp_[j] = lmnp[lmnp_idx];
                    }
                } // end for j

                __syncthreads();

                // Iterate batch
                for (unsigned int j = 0; j < BATCH_SIZE_PIXELS; j++) {
                    // Load l,m,n and phase_offset
                    float l = lmnp_[j].x;
                    float m = lmnp_[j].y;
                    float n = lmnp_[j].z;
                    float phase_offset = lmnp_[j].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                    // Update sum
                    for (unsigned int term_nr = 0; term_nr < MAX_NR_TERMS; term_nr++) {
                        float4 a = pixels_[term_nr][j][0];
                        float4 b = pixels_[term_nr][j][1];
                        float2 pixel_xx = make_float2(a.x, a.y);
                        float2 pixel_xy = make_float2(a.z, a.w);
                        float2 pixel_yx = make_float2(b.x, b.y);
                        float2 pixel_yy = make_float2(b.z, b.w);
                        cmac(sum[term_nr][0], phasor, pixel_xx);
                        cmac(sum[term_nr][1], phasor, pixel_xy);
                        cmac(sum[term_nr][2], phasor, pixel_yx);
                        cmac(sum[term_nr][3], phasor, pixel_yy);
                    } // end for term_nr
                } // end for j (batch)
            } // end for pixel_offset

            const float scale = 1.0f / nr_pixels;
            if (time_idx_batch < current_nr_timesteps) {
                for (unsigned int term_nr = 0; term_nr < MAX_NR_TERMS; term_nr++) {
                    for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned int sum_idx = index_sums(total_nr_timesteps, nr_channels, term_nr, pol, time_idx_global, chan_idx_local);
                        sums[sum_idx] = sum[term_nr][pol] * scale;
                    } // end for pol
                } // end for term_nr
            } // end if time

        } // end for i (visibilities)
    } // end for time_offset_local
} // end kernel_calibrate_sums


__global__ void kernel_calibrate_gradient(
    const int                      subgrid_size,
    const float                    image_size,
    const int                      total_nr_timesteps,
    const int                      nr_channels,
    const int                      nr_stations,
    const int                      term_offset,
    const int                      current_nr_terms,
    const int                      nr_terms,
    const UVW<float>* __restrict__ uvw,
    const float*      __restrict__ wavenumbers,
    const float2*     __restrict__ visibilities,
    const float*      __restrict__ weights,
    const float2*     __restrict__ aterm,
    const float2*     __restrict__ aterm_derivatives,
    const int*        __restrict__ aterm_indices,
    const Metadata*   __restrict__ metadata,
    const float2*     __restrict__ subgrid,
    const float2*     __restrict__ sums,
    const float4*     __restrict__ lmnp,
          double*     __restrict__ gradient,
          double*     __restrict__ residual_sum)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Load metadata for first subgrid
    const Metadata &m0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset_global = m.time_index - m0.time_index;
    const unsigned int station1           = m.baseline.station1;
    const unsigned int station2           = m.baseline.station2;
    const unsigned int nr_timesteps       = m.nr_timesteps;
    const unsigned int channel_begin      = m.channel_begin;
    const unsigned int channel_end        = m.channel_end;
    const unsigned int nr_channels_local  = channel_end - channel_begin;

    // Shared memory
    __shared__ float2 pixels_[NR_POLARIZATIONS][BATCH_SIZE_PIXELS];
    __shared__ float4 lmnp_[BATCH_SIZE_PIXELS];

    // Accumulate gradient update in registers
    double update[MAX_NR_TERMS];
    double update_residual_sum;

    // Iterate timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        int aterm_idx = aterm_indices[time_offset_global + time_offset_local];

        // Reset update to zero
        for (unsigned int term_nr = 0; term_nr < MAX_NR_TERMS; term_nr++) {
            update[term_nr] = 0.0;
        }
        update_residual_sum = 0.0;

        // Determine number of timesteps to process
        current_nr_timesteps = 0;
        for (int time = time_offset_local; time < nr_timesteps; time++) {
            if (aterm_indices[time_offset_global + time] == aterm_idx) {
                current_nr_timesteps++;
            } else {
                break;
            }
        }

        // Iterate batch of visibilities from the same timeslot
        for (int i = tid; i < ALIGN(current_nr_timesteps * nr_channels_local, nr_threads); i += nr_threads) {
            unsigned int time_idx_batch  = (i / nr_channels_local);
            unsigned int chan_idx_local  = (i % nr_channels_local) + channel_begin;
            unsigned int time_idx_local  = time_offset_local + time_idx_batch;
            unsigned int time_idx_global = time_offset_global + time_idx_local;

            // Load UVW
            float u, v, w;
            if (time_idx_batch < current_nr_timesteps) {
                u = uvw[time_idx_global].u;
                v = uvw[time_idx_global].v;
                w = uvw[time_idx_global].w;
            }

            // Load wavenumber
            float wavenumber = wavenumbers[chan_idx_local];

            // Accumulate sums in registers
            float2 sum[NR_POLARIZATIONS] = {0, 0};

            // Iterate all pixels
            for (unsigned int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += BATCH_SIZE_PIXELS) {

                __syncthreads();

                for (unsigned int j = tid; j < BATCH_SIZE_PIXELS; j += nr_threads) {
                    unsigned int y = (pixel_offset + j) / subgrid_size;
                    unsigned int x = (pixel_offset + j) % subgrid_size;

                    // Reset pixel to zero
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixels_[pol][j] = make_float2(0, 0);
                    }

                    // Prepare batch
                    if (y < subgrid_size) {
                        // Compute shifted position in subgrid
                        unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                        unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                        // Load pixel and aterms
                        float2 pixel[NR_POLARIZATIONS];
                        float2 aterm1[NR_POLARIZATIONS];
                        float2 aterm2[NR_POLARIZATIONS];
                        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            unsigned int pixel_idx  = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                            unsigned int aterm1_idx = index_aterm_transposed(subgrid_size, nr_stations, aterm_idx, station1, y, x, pol);
                            unsigned int aterm2_idx = index_aterm_transposed(subgrid_size, nr_stations, aterm_idx, station2, y, x, pol);
                            pixel[pol]  = subgrid[pixel_idx];
                            aterm1[pol] = aterm[aterm1_idx];
                            aterm2[pol] = aterm[aterm2_idx];
                        }

                        // Apply aterm
                        apply_aterm_degridder(pixel, aterm1, aterm2);

                        // Store pixel in shared memory
                        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            pixels_[pol][j] = pixel[pol];
                        }

                        // Load l,m,n and phase_offset into shared memory
                        unsigned int lmnp_idx = index_lmnp(subgrid_size, s, y, x);
                        lmnp_[j] = lmnp[lmnp_idx];
                    } // end if
                } // end for j

                __syncthreads();

                // Iterate batch
                for (unsigned int j = 0; j < BATCH_SIZE_PIXELS; j++) {
                    // Load l,m,n and phase_offset
                    float l = lmnp_[j].x;
                    float m = lmnp_[j].y;
                    float n = lmnp_[j].z;
                    float phase_offset = lmnp_[j].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                    // Update sum
                    for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        cmac(sum[pol], phasor, pixels_[pol][j]);
                    }
                } // end for j (batch)
            } // end for pixel_offset

            if (time_idx_batch < current_nr_timesteps) {

                // Compute residual
                float2 residual[NR_POLARIZATIONS];
                float2 residual_weighted[NR_POLARIZATIONS];
                const float scale = 1.0f / nr_pixels;
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    unsigned int vis_idx = index_visibility(nr_channels, time_idx_global, chan_idx_local, pol);
                    residual[pol] = visibilities[vis_idx] - (sum[pol] * scale);
                    residual_weighted[pol] = residual[pol] * weights[vis_idx];

                    // Compute gradient update
                    for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
                        unsigned int sum_idx = index_sums(total_nr_timesteps, nr_channels, term_nr, pol, time_idx_global, chan_idx_local);
                        update[term_nr] += residual_weighted[pol].x * sums[sum_idx].x;
                        update[term_nr] += residual_weighted[pol].y * sums[sum_idx].y;
                    } // end for term

                    // Compute residual_sum update
                    update_residual_sum += residual_weighted[pol].x*residual[pol].x;
                    update_residual_sum += residual_weighted[pol].y*residual[pol].y;

                } // end for pol
            } // end if time
        } // end for i (visibilities)

        // Update gradient
        for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
            unsigned int idx = aterm_idx * nr_terms + (term_offset + term_nr);
            atomicAdd(&gradient[idx], update[term_nr]);
        }

        if (!term_offset) atomicAdd(residual_sum, update_residual_sum);

    } // end for time_offset_local
} // end kernel_calibrate_gradient


__global__ void kernel_calibrate_hessian(
    const int                    total_nr_timesteps,
    const int                    nr_channels,
    const int                    term_offset_y,
    const int                    term_offset_x,
    const int                    nr_terms,
    const float*    __restrict__ weights,
    const int*      __restrict__ aterm_indices,
    const Metadata* __restrict__ metadata,
    const float2*   __restrict__ sums_y,
    const float2*   __restrict__ sums_x,
          double*   __restrict__ hessian)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned s          = blockIdx.x;

    // Metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset_global = m.time_index;
    const unsigned int nr_timesteps       = m.nr_timesteps;
    const unsigned int channel_begin      = m.channel_begin;
    const unsigned int channel_end        = m.channel_end;
    const unsigned int nr_channels_local  = channel_end - channel_begin;

    // Iterate timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        int aterm_idx = aterm_indices[time_offset_global + time_offset_local];

        // Determine number of timesteps to process
        current_nr_timesteps = 0;
        for (int time = time_offset_local; time < nr_timesteps; time++) {
            if (aterm_indices[time_offset_global + time] == aterm_idx) {
                current_nr_timesteps++;
            } else {
                break;
            }
        }

        // Set term nubmers
        unsigned int term_nr1 = tidx;
        unsigned int term_nr0 = tidy;

        // Compute hessian update
        double update = 0.0;

        // Iterate all timesteps
        for (unsigned int time = 0; time < current_nr_timesteps; time++) {

            // Iterate all channels
            for (unsigned int chan = 0; chan < nr_channels_local; chan++) {

                // Iterate all polarizations
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    unsigned int time_idx_global = time_offset_global + time_offset_local + time;
                    unsigned int chan_idx_local  = channel_begin + chan;
                    unsigned int  vis_idx = index_visibility(nr_channels, time_idx_global, chan_idx_local, pol);
                    unsigned int sum_idx0 = index_sums(total_nr_timesteps, nr_channels, term_nr0, pol, time_idx_global, chan_idx_local);
                    unsigned int sum_idx1 = index_sums(total_nr_timesteps, nr_channels, term_nr1, pol, time_idx_global, chan_idx_local);
                    float2 sum0 = sums_y[sum_idx0];
                    float2 sum1 = sums_x[sum_idx1] * weights[vis_idx];

                    // Update hessian
                    update += sum0.x * sum1.x + sum0.y * sum1.y;
                } // end for pol
            } // end chan
        } // end for time

        // Compute term indices
        unsigned int term_idx1 = term_offset_x + term_nr1;
        unsigned int term_idx0 = term_offset_y + term_nr0;

        // Update hessian
        unsigned int idx = aterm_idx * nr_terms * nr_terms +
                           term_idx1 * nr_terms + term_idx0;
        atomicAdd(&hessian[idx], update);

        // Update mirror hessian
        if (term_offset_y != term_offset_x) {
            unsigned int idx = aterm_idx * nr_terms * nr_terms +
                               term_idx0 * nr_terms + term_idx1;
            atomicAdd(&hessian[idx], update);
        }
    } // end for time_offset_local
} // end kernel_calibrate_hessian

} // end extern "C"
