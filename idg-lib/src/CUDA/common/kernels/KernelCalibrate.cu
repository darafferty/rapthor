#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS     8
#define BATCH_SIZE_SUMS     256
#define BATCH_SIZE_GRADIENT 128

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

inline __device__ float4 compute_lmnp(
    const Coordinate coordinate,
    const int        y,
    const int        x,
    const int        grid_size,
    const int        subgrid_size,
    const float      image_size,
    const float      w_step)
{
    // Locateion of current subgrid
    const int x_coordinate = coordinate.x;
    const int y_coordinate = coordinate.y;
    const int z_coordinate = coordinate.z;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) z_coordinate + 0.5) * 2 * M_PI;

    // Compute l,m,n and phase_offset
    float l = compute_l(x, subgrid_size, image_size);
    float m = compute_m(y, subgrid_size, image_size);
    float n = compute_n(l, m);
    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

    // Return result
    return make_float4(l, m, n, phase_offset);
} // end compute_lmnp


template<int current_nr_terms>
__device__ void update_sums(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const unsigned int                total_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_stations,
    const unsigned int                nr_terms,
    const unsigned int                term_offset,
    const UVW<float>*    __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const int*           __restrict__ aterm_indices,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sums,
          float4*                     lmnp_,
          float2                      pixels_[NR_POLARIZATIONS][BATCH_SIZE_SUMS])
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
    const unsigned int station2     = m.baseline.station2;
    const unsigned int nr_timesteps = m.nr_timesteps;
    const Coordinate coordinate     = m.coordinate;

    // Iterate all terms
    for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {

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
            for (int i = tid; i < ALIGN(current_nr_timesteps * nr_channels, nr_threads); i += nr_threads) {
                int time = (i / nr_channels) + time_offset_local;
                int chan = (i % nr_channels);

                // Load UVW
                float u, v, w;
                if (time < nr_timesteps) {
                    u = uvw[time_offset_global + time].u;
                    v = uvw[time_offset_global + time].v;
                    w = uvw[time_offset_global + time].w;
                }

                // Load wavenumber
                float wavenumber = wavenumbers[chan];

                // Accumulate sums in registers
                float2 sum_xx = make_float2(0, 0);
                float2 sum_xy = make_float2(0, 0);
                float2 sum_yx = make_float2(0, 0);
                float2 sum_yy = make_float2(0, 0);

                // Iterate all pixels
                for (unsigned int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += BATCH_SIZE_SUMS) {
                    __syncthreads();

                    float2 pixel[NR_POLARIZATIONS];
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixel[pol] = make_float2(0, 0);
                    }

                    for (unsigned int j = tid; j < BATCH_SIZE_SUMS; j += nr_threads) {
                        unsigned int y = (pixel_offset + j) / subgrid_size;
                        unsigned int x = (pixel_offset + j) % subgrid_size;

                        // Prepare batch
                        if (y < subgrid_size) {
                            // Compute shifted position in subgrid
                            unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                            unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                            // Load pixels and aterms
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
                            apply_aterm_calibrate(pixel, aterm1, aterm2);

                            // Store pixels in shared memory
                            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                pixels_[pol][j] = pixel[pol];
                            }

                            lmnp_[j] = compute_lmnp(coordinate, y, x, grid_size, subgrid_size, image_size, w_step);
                        }
                    }

                    __syncthreads();

                    // Iterate batch
                    for (unsigned int j = 0; j < BATCH_SIZE_SUMS; j++) {
                        // Load l,m,n
                        float l = lmnp_[j].x;
                        float m = lmnp_[j].y;
                        float n = lmnp_[j].z;

                        // Load phase offset
                        float phase_offset = lmnp_[j].w;

                        // Compute phase index
                        float phase_index = u*l + v*m + w*n;

                        // Compute phasor
                        float  phase  = (phase_index * wavenumber) - phase_offset;
                        float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                        // Load pixels
                        float2 pixel_xx = pixels_[0][j];
                        float2 pixel_xy = pixels_[1][j];
                        float2 pixel_yx = pixels_[2][j];
                        float2 pixel_yy = pixels_[3][j];

                        // Update sums
                        sum_xx.x += phasor.x * pixel_xx.x;
                        sum_xx.y += phasor.x * pixel_xx.y;
                        sum_xx.x -= phasor.y * pixel_xx.y;
                        sum_xx.y += phasor.y * pixel_xx.x;

                        sum_xy.x += phasor.x * pixel_xy.x;
                        sum_xy.y += phasor.x * pixel_xy.y;
                        sum_xy.x -= phasor.y * pixel_xy.y;
                        sum_xy.y += phasor.y * pixel_xy.x;

                        sum_yx.x += phasor.x * pixel_yx.x;
                        sum_yx.y += phasor.x * pixel_yx.y;
                        sum_yx.x -= phasor.y * pixel_yx.y;
                        sum_yx.y += phasor.y * pixel_yx.x;

                        sum_yy.x += phasor.x * pixel_yy.x;
                        sum_yy.y += phasor.x * pixel_yy.y;
                        sum_yy.x -= phasor.y * pixel_yy.y;
                        sum_yy.y += phasor.y * pixel_yy.x;
                    } // end for k (batch)
                } // end for j (pixels)

                const float scale = 1.0f / nr_pixels;
                if (time < nr_timesteps) {
                    unsigned int time_idx = time_offset_global + time;
                    unsigned int chan_idx = chan;
                    unsigned int sum_idx_xx = index_sums(total_nr_timesteps, nr_channels, term_nr, 0, time_idx, chan_idx);
                    unsigned int sum_idx_xy = index_sums(total_nr_timesteps, nr_channels, term_nr, 1, time_idx, chan_idx);
                    unsigned int sum_idx_yx = index_sums(total_nr_timesteps, nr_channels, term_nr, 2, time_idx, chan_idx);
                    unsigned int sum_idx_yy = index_sums(total_nr_timesteps, nr_channels, term_nr, 3, time_idx, chan_idx);
                    sums[sum_idx_xx] = sum_xx * scale;
                    sums[sum_idx_xy] = sum_xy * scale;
                    sums[sum_idx_yx] = sum_yx * scale;
                    sums[sum_idx_yy] = sum_yy * scale;
                }

                __syncthreads();

            } // end for i (visibilities)
        } // end for time_offset_local
    } // end for term_nr
} // end update_sums


__device__ void update_gradient(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const unsigned int                total_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_stations,
    const unsigned int                nr_terms,
    const UVW<float>*    __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const int*           __restrict__ aterm_indices,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sums,
          float2*        __restrict__ gradient)
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
    const unsigned int nr_timesteps = m.nr_timesteps;
    const unsigned int station1     = m.baseline.station1;
    const unsigned int station2     = m.baseline.station2;
    const Coordinate coordinate     = m.coordinate;

    // Shared memory
    __shared__ float4 lmnp_[BATCH_SIZE_GRADIENT];
    __shared__ float2 pixels_[NR_POLARIZATIONS][BATCH_SIZE_GRADIENT];

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
        for (int i = tid; i < ALIGN(current_nr_timesteps * nr_channels, nr_threads); i += nr_threads) {
            int time = (i / nr_channels) + time_offset_local;
            int chan = (i % nr_channels);

            // Load UVW
            float u, v, w;
            if (time < nr_timesteps) {
                u = uvw[time_offset_global + time].u;
                v = uvw[time_offset_global + time].v;
                w = uvw[time_offset_global + time].w;
            }

            // Load wavenumber
            float wavenumber = wavenumbers[chan];

            // Accumulate sums in registers
            float2 sum_xx = make_float2(0, 0);
            float2 sum_xy = make_float2(0, 0);
            float2 sum_yx = make_float2(0, 0);
            float2 sum_yy = make_float2(0, 0);

            // Iterate all pixels
            for (unsigned int j = tid; j < ALIGN(nr_pixels, nr_threads); j += nr_threads) {
                unsigned int y = j / subgrid_size;
                unsigned int x = j % subgrid_size;

                __syncthreads();

                // Prepare batch
                if (j < nr_pixels) {
                    // Compute shifted position in subgrid
                    unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                    unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                    // Load pixels and aterms
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
                    apply_aterm_calibrate(pixel, aterm1, aterm2);

                    // Store pixels in shared memory
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixels_[pol][tid] = pixel[pol];
                    }

                    lmnp_[tid] = compute_lmnp(coordinate, y, x, grid_size, subgrid_size, image_size, w_step);
                }

                __syncthreads();

                // Iterate batch
                for (unsigned int k = 0; k < nr_threads; k++) {
                    unsigned int pixel_idx = (j - tid) + k;

                    if (pixel_idx < nr_pixels) {
                        // Load l,m,n
                        float l = lmnp_[k].x;
                        float m = lmnp_[k].y;
                        float n = lmnp_[k].z;

                        // Load phase offset
                        float phase_offset = lmnp_[k].w;

                        // Compute phase index
                        float phase_index = u*l + v*m + w*n;

                        // Compute phasor
                        float  phase  = (phase_index * wavenumber) - phase_offset;
                        float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                        // Load pixels
                        float2 pixel_xx = pixels_[0][k];
                        float2 pixel_xy = pixels_[1][k];
                        float2 pixel_yx = pixels_[2][k];
                        float2 pixel_yy = pixels_[3][k];

                        // Update sums
                        sum_xx.x += phasor.x * pixel_xx.x;
                        sum_xx.y += phasor.x * pixel_xx.y;
                        sum_xx.x -= phasor.y * pixel_xx.y;
                        sum_xx.y += phasor.y * pixel_xx.x;

                        sum_xy.x += phasor.x * pixel_xy.x;
                        sum_xy.y += phasor.x * pixel_xy.y;
                        sum_xy.x -= phasor.y * pixel_xy.y;
                        sum_xy.y += phasor.y * pixel_xy.x;

                        sum_yx.x += phasor.x * pixel_yx.x;
                        sum_yx.y += phasor.x * pixel_yx.y;
                        sum_yx.x -= phasor.y * pixel_yx.y;
                        sum_yx.y += phasor.y * pixel_yx.x;

                        sum_yy.x += phasor.x * pixel_yy.x;
                        sum_yy.y += phasor.x * pixel_yy.y;
                        sum_yy.x -= phasor.y * pixel_yy.y;
                        sum_yy.y += phasor.y * pixel_yy.x;
                    } // end if
                } // end for k (batch)
            } // end for j (pixels)

            // Compute sums
            const float scale = 1.0f / nr_pixels;

            if ((time - time_offset_local) < current_nr_timesteps) {

                // Iterate all terms
                for (unsigned term_nr = 0; term_nr < nr_terms; term_nr++) {

                    // Compute residual
                    unsigned int time_idx = time_offset_global + time;
                    unsigned int chan_idx = chan;
                    unsigned int vis_idx_xx = index_visibility(nr_channels, time_idx, chan_idx, 0);
                    unsigned int vis_idx_xy = index_visibility(nr_channels, time_idx, chan_idx, 1);
                    unsigned int vis_idx_yx = index_visibility(nr_channels, time_idx, chan_idx, 2);
                    unsigned int vis_idx_yy = index_visibility(nr_channels, time_idx, chan_idx, 3);
                    float2 residual_xx = (visibilities[vis_idx_xx] - (sum_xx * scale)) * weights[vis_idx_xx];
                    float2 residual_xy = (visibilities[vis_idx_xy] - (sum_xy * scale)) * weights[vis_idx_xy];
                    float2 residual_yx = (visibilities[vis_idx_yx] - (sum_yx * scale)) * weights[vis_idx_yx];
                    float2 residual_yy = (visibilities[vis_idx_yy] - (sum_yy * scale)) * weights[vis_idx_yy];

                    // Load derivative sums
                    unsigned int sums_idx_xx = index_sums(total_nr_timesteps, nr_channels, term_nr, 0, time_idx, chan_idx);
                    unsigned int sums_idx_xy = index_sums(total_nr_timesteps, nr_channels, term_nr, 1, time_idx, chan_idx);
                    unsigned int sums_idx_yx = index_sums(total_nr_timesteps, nr_channels, term_nr, 2, time_idx, chan_idx);
                    unsigned int sums_idx_yy = index_sums(total_nr_timesteps, nr_channels, term_nr, 3, time_idx, chan_idx);
                    float2 sum_xx = sums[sums_idx_xx];
                    float2 sum_xy = sums[sums_idx_xy];
                    float2 sum_yx = sums[sums_idx_yx];
                    float2 sum_yy = sums[sums_idx_yy];

                    // Compute gradient update
                    float2 update = make_float2(0, 0);

                    update.x += sum_xx.x * residual_xx.x;
                    update.x += sum_xx.y * residual_xx.y;
                    update.y += sum_xx.x * residual_xx.y;
                    update.y -= sum_xx.y * residual_xx.x;

                    update.x += sum_xy.x * residual_xy.x;
                    update.x += sum_xy.y * residual_xy.y;
                    update.y += sum_xy.x * residual_xy.y;
                    update.y -= sum_xy.y * residual_xy.x;

                    update.x += sum_yx.x * residual_yx.x;
                    update.x += sum_yx.y * residual_yx.y;
                    update.y += sum_yx.x * residual_yx.y;
                    update.y -= sum_yx.y * residual_yx.x;

                    update.x += sum_yy.x * residual_yy.x;
                    update.x += sum_yy.y * residual_yy.y;
                    update.y += sum_yy.x * residual_yy.y;
                    update.y -= sum_yy.y * residual_yy.x;

                    // Update gradient
                    unsigned int idx = aterm_idx * nr_terms + term_nr;
                    atomicAdd(&gradient[idx], update);

                } // end for term
            } // end if time

            __syncthreads();

        } // end for i (visibilities)
    }  // end for time_offset_local
} // end update_gradient


__device__ void update_hessian(
    const unsigned int                total_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_stations,
    const unsigned int                nr_terms,
    const int*           __restrict__ aterm_indices,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ sums,
          float2*        __restrict__ hessian)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned s          = blockIdx.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    // Metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset_global = m.time_index;
    const unsigned int nr_timesteps = m.nr_timesteps;

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

        // Iterate all terms * terms
        for (unsigned int term_nr = tid; term_nr < (nr_terms*nr_terms); term_nr += nr_threads) {
            unsigned term_nr1 = term_nr / nr_terms;
            unsigned term_nr0 = term_nr % nr_terms;

            // Compute hessian update
            float2 update = make_float2(0, 0);

            // Iterate all timesteps
            for (unsigned int time = 0; time < current_nr_timesteps; time++) {

                // Iterate all channels
                for (unsigned int chan = 0; chan < nr_channels; chan++) {

                    // Iterate all polarizations
                    for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned int time_idx = time_offset_global + time_offset_local + time;
                        unsigned int chan_idx = chan;
                        unsigned int  vis_idx = index_visibility(nr_channels, time_idx, chan_idx, pol);
                        unsigned int sum_idx0 = index_sums(total_nr_timesteps, nr_channels, term_nr0, pol, time_idx, chan_idx);
                        unsigned int sum_idx1 = index_sums(total_nr_timesteps, nr_channels, term_nr1, pol, time_idx, chan_idx);
                        float2 sum0 = sums[sum_idx0];
                        float2 sum1 = sums[sum_idx1] * weights[vis_idx];

                        // Update hessian
                        if (term_nr0 < nr_terms) {
                            update.x += sum1.x * sum0.x;
                            update.x += sum1.y * sum0.y;
                            update.y += sum1.y * sum0.x;
                            update.y -= sum1.x * sum0.y;
                        }
                    } // end for pol
                } // end chan
            } // end for time

            __syncthreads();

            // Update hessian
            if (term_nr0 < nr_terms) {
                unsigned idx = aterm_idx * nr_terms * nr_terms + term_nr1 * nr_terms + term_nr0;
                atomicAdd(&hessian[idx], update);
            }
        } // end for term_nr (terms * terms)
    } // end for time_offset_local
} // end update_hessian


#define UPDATE_SUMS(current_nr_terms) \
    for (; (term_offset + current_nr_terms) <= nr_terms; term_offset += current_nr_terms) { \
        update_sums<current_nr_terms>( \
                grid_size, subgrid_size, image_size, w_step, total_nr_timesteps, nr_channels, nr_stations, \
                nr_terms, term_offset, \
                uvw, aterm, aterm_indices, aterm_derivatives, wavenumbers, \
                visibilities, weights, metadata, subgrid, \
                sums, lmnp_, pixels_); \
    }

extern "C" {

    __shared__ float4 lmnp_[BATCH_SIZE_SUMS];
    __shared__ float2 pixels_[MAX_NR_TERMS][NR_POLARIZATIONS][BATCH_SIZE_SUMS];
__global__ void kernel_calibrate_sums(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         total_nr_timesteps,
    const int                         nr_channels,
    const int                         nr_stations,
    const int                         nr_terms,
    const UVW<float>*    __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const int*           __restrict__ aterm_indices,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sums,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    // Shared memory
    __shared__ float4 lmnp_[BATCH_SIZE_SUMS];
    __shared__ float2 pixels_[NR_POLARIZATIONS][BATCH_SIZE_SUMS];

    int term_offset = 0;
    UPDATE_SUMS(8)
    UPDATE_SUMS(7)
    UPDATE_SUMS(6)
    UPDATE_SUMS(5)
    UPDATE_SUMS(4)
    UPDATE_SUMS(3)
    UPDATE_SUMS(2)
    UPDATE_SUMS(1)
} // end kernel_calibrate_sums


__global__ void kernel_calibrate_gradient(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         total_nr_timesteps,
    const int                         nr_channels,
    const int                         nr_stations,
    const int                         nr_terms,
    const UVW<float>*    __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const int*           __restrict__ aterm_indices,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sums,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    update_gradient(
        grid_size, subgrid_size, image_size, w_step, total_nr_timesteps,
        nr_channels, nr_stations, nr_terms,
        uvw, aterm, aterm_indices, aterm_derivatives,
        wavenumbers, visibilities, weights, metadata, subgrid,
        sums, gradient);
} // end kernel_calibrate_gradient


__global__ void kernel_calibrate_hessian(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         total_nr_timesteps,
    const int                         nr_channels,
    const int                         nr_stations,
    const int                         nr_terms,
    const UVW<float>*    __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const int*           __restrict__ aterm_indices,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sums,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    update_hessian(
        total_nr_timesteps, nr_channels, nr_stations, nr_terms,
        aterm_indices, visibilities, weights, metadata, sums, hessian);
} // end kernel_calibrate_hessian

} // end extern "C"
