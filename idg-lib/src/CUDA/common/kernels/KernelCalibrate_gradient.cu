#include "Types.h"
#include "math.cu"

#include "KernelCalibrate_index.cuh"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS      8
#define BATCH_SIZE_PIXELS 64

extern "C" {

__global__ void kernel_calibrate_gradient(
    const int                        nr_polarizations,
    const int                        subgrid_size,
    const float                      image_size,
    const int                        total_nr_timesteps,
    const int                        nr_channels,
    const int                        nr_stations,
    const int                        term_offset,
    const int                        current_nr_terms,
    const int                        nr_terms,
    const UVW<float>*   __restrict__ uvw,
    const float*        __restrict__ wavenumbers,
    const float2*       __restrict__ visibilities,
    const float*        __restrict__ weights,
    const float2*       __restrict__ aterm,
    const unsigned int* __restrict__ aterm_indices,
    const Metadata*     __restrict__ metadata,
    const float2*       __restrict__ subgrid,
    const float2*       __restrict__ sums, // derivatives of the visibilities
    const float4*       __restrict__ lmnp,
          double*       __restrict__ gradient,
          double*       __restrict__ residual_sum)
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
    __shared__ float2 pixels_[4][BATCH_SIZE_PIXELS];
    __shared__ float4 lmnp_[BATCH_SIZE_PIXELS];

    // Accumulate gradient update in registers
    double update[MAX_NR_TERMS];
    double update_residual_sum;

    // Iterate timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        const unsigned int aterm_idx = aterm_indices[time_offset_global + time_offset_local];

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
            float2 sum[4] = {0, 0};

            // Iterate all pixels
            for (unsigned int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += BATCH_SIZE_PIXELS) {

                __syncthreads();

                for (unsigned int j = tid; j < BATCH_SIZE_PIXELS; j += nr_threads) {
                    unsigned int y = (pixel_offset + j) / subgrid_size;
                    unsigned int x = (pixel_offset + j) % subgrid_size;

                    // Reset pixel to zero
                    for (unsigned pol = 0; pol < nr_polarizations; pol++) {
                        pixels_[pol][j] = make_float2(0, 0);
                    }

                    // Prepare batch
                    if (y < subgrid_size) {
                        // Compute shifted position in subgrid
                        unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                        unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                        // Load pixel and aterms
                        float2 pixel[4];
                        float2 aterm1[4];
                        float2 aterm2[4];
                        for (unsigned pol = 0; pol < nr_polarizations; pol++) {
                            unsigned int pixel_idx  = index_subgrid(4, subgrid_size, s, pol, y_src, x_src);
                            unsigned int aterm1_idx = index_aterm_transposed(4, subgrid_size, nr_stations, aterm_idx, station1, y, x, pol);
                            unsigned int aterm2_idx = index_aterm_transposed(4, subgrid_size, nr_stations, aterm_idx, station2, y, x, pol);
                            pixel[pol]  = subgrid[pixel_idx];
                            aterm1[pol] = aterm[aterm1_idx];
                            aterm2[pol] = aterm[aterm2_idx];
                        }

                        // Apply aterm
                        apply_aterm_degridder(pixel, aterm1, aterm2);

                        // Store pixel in shared memory
                        for (unsigned pol = 0; pol < nr_polarizations; pol++) {
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
                    float phase = (phase_index * wavenumber) - phase_offset;
                    float2 phasor;
                    __sincosf(phase, &phasor.y, &phasor.x);

                    // Update sum
                    for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
                        cmac(sum[pol], phasor, pixels_[pol][j]);
                    }
                } // end for j (batch)
            } // end for pixel_offset

            if (time_idx_batch < current_nr_timesteps) {

                // Compute residual
                float2 residual[4];
                float2 residual_weighted[4];
                const float scale = 1.0f / nr_pixels;
                for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
                    unsigned int vis_idx = index_visibility(nr_polarizations, nr_channels, time_idx_global, chan_idx_local, pol);
                    const float2 model_visibility = sum[pol] * scale;
                    residual[pol] = visibilities[vis_idx] - model_visibility;
                    residual_weighted[pol] = residual[pol] * weights[vis_idx];

                    // Compute gradient update
                    for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
                        unsigned int sum_idx = index_sums(4, total_nr_timesteps, nr_channels, term_nr, pol, time_idx_global, chan_idx_local);
                        const float2 visibility_derivative = sums[sum_idx];
                        update[term_nr] += residual_weighted[pol].x * visibility_derivative.x;
                        update[term_nr] += residual_weighted[pol].y * visibility_derivative.y;
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

} // end extern "C"
