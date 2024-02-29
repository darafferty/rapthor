#include "Types.h"
#include "math.cu"

#include "KernelCalibrate_index.cuh"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS      8
#define BATCH_SIZE_PIXELS 64

extern "C" {

__global__ void kernel_calibrate_sums(
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
    const float2*       __restrict__ aterm,
    const float2*       __restrict__ aterm_derivatives,
    const unsigned int* __restrict__ aterm_indices,
    const Metadata*     __restrict__ metadata,
    const float2*       __restrict__ subgrid,
          float2*       __restrict__ sums,
    const float4*       __restrict__ lmnp)
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
        const unsigned int aterm_idx = aterm_indices[time_offset_global + time_offset_local];

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
            float2 sum[MAX_NR_TERMS][4] = {0, 0};

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
                            float2 pixel[4];
                            float2 aterm1[4];
                            float2 aterm2[4];
                            for (unsigned pol = 0; pol < nr_polarizations; pol++) {
                                unsigned int term_idx   = term_offset + term_nr;
                                unsigned int pixel_idx  = index_subgrid(4, subgrid_size, s, pol, y_src, x_src);
                                unsigned int aterm1_idx = index_aterm_transposed(4, subgrid_size, nr_terms, aterm_idx, term_idx, y, x, pol);
                                unsigned int aterm2_idx = index_aterm_transposed(4, subgrid_size, nr_stations, aterm_idx, station2, y, x, pol);
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
                    float2 phasor;
                    __sincosf(phase, &phasor.y, &phasor.x);

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
                    for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
                        unsigned int sum_idx = index_sums(nr_polarizations, total_nr_timesteps, nr_channels, term_nr, pol, time_idx_global, chan_idx_local);
                        sums[sum_idx] = sum[term_nr][pol] * scale;
                    } // end for pol
                } // end for term_nr
            } // end if time

        } // end for i (visibilities)
    } // end for time_offset_local
} // end kernel_calibrate_sums

} // end extern "C"
