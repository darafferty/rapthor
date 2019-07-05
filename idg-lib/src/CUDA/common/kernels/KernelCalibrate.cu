#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS     8
#define MAX_SUBGRID_SIZE 32
#define MAX_NR_THREADS   128
#define MAX_NR_TIMESTEPS 128

inline __device__ long index_sum_deriv(
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

inline __device__ long index_sum_aterm(
    unsigned int total_nr_timesteps, // number of timesteps for all baselines
    unsigned int nr_channels,        // number channels for a single baseline
    unsigned int pol,
    unsigned int time,
    unsigned int chan)
{
    // sums: [NR_POLARIZATIONS][TOTAL_NR_TIMESTEPS][NR_CHANNELS]
    return pol * total_nr_timesteps * nr_channels +
           time * nr_channels +
           chan;
}

__device__ void compute_lmnp(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const Metadata*      __restrict__ metadata,
          float4                      lmnp_[MAX_SUBGRID_SIZE][MAX_SUBGRID_SIZE])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int x_coordinate      = m.coordinate.x;
    const int y_coordinate      = m.coordinate.y;
    const int z_coordinate      = m.coordinate.z;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) z_coordinate + 0.5) * 2 * M_PI;

    for (unsigned int i = tid; i < (subgrid_size * subgrid_size); i += nr_threads) {
        unsigned int y = i / subgrid_size;
        unsigned int x = i % subgrid_size;

        if (y < subgrid_size) {
            float l = compute_l(x, subgrid_size, image_size);
            float m = compute_m(y, subgrid_size, image_size);
            float n = compute_n(l, m);
            float phase_offset = u_offset*l + v_offset*m + w_offset*n;
            lmnp_[y][x] = make_float4(l, m, n, phase_offset);
        }
    }
} // end compute_lmnp


/*
    This method has two modes:
        UPDATE_MODE_DERIV:
            Compute sums for the current station by combining aterm_derivatives
            with the aterm for station2 and storing the result in sum_deriv
        UPDATE_MODE_ATERM:
            Compute sums for every visibility by combining the aterms
            for station1 and station 2 and storing the result in sum_aterm
*/
#define UPDATE_MODE_DERIV 0
#define UPDATE_MODE_ATERM 1
template<int current_nr_terms, int mode>
__device__ void update_sums(
    const int                         subgrid_size,
    const float                       image_size,
    const unsigned int                total_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_stations,
    const unsigned int                nr_terms,
    const unsigned int                term_offset,
    const UVW*           __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const int*           __restrict__ aterm_indices,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sum_aterm,
          float2*        __restrict__ sum_deriv,
          float4                      lmnp_[MAX_SUBGRID_SIZE][MAX_SUBGRID_SIZE],
          float2                      pixels_[MAX_NR_TERMS][NR_POLARIZATIONS][MAX_SUBGRID_SIZE])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    if (mode == UPDATE_MODE_ATERM) {
        assert(current_nr_terms == 1);
    }

    // Load metadata for first subgrid
    const Metadata &m0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset_global = m.time_index - m0.time_index;
    const unsigned int station1     = m.baseline.station1;
    const unsigned int station2     = m.baseline.station2;
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
            float2 sum_xx[current_nr_terms];
            float2 sum_xy[current_nr_terms];
            float2 sum_yx[current_nr_terms];
            float2 sum_yy[current_nr_terms];
            for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
                sum_xx[term_nr] = make_float2(0, 0);
                sum_xy[term_nr] = make_float2(0, 0);
                sum_yx[term_nr] = make_float2(0, 0);
                sum_yy[term_nr] = make_float2(0, 0);
            }

            // Iterate all rows of the subgrid
            for (unsigned int y = 0; y < subgrid_size; y++) {
                __syncthreads();

                // Precompute data for one row
                for (unsigned j = tid; j < (subgrid_size*nr_terms); j += nr_threads) {
                    unsigned int term_nr = j / subgrid_size;
                    unsigned int x       = j % subgrid_size;

                    if (term_nr < nr_terms) {
                        // Compute shifted position in subgrid
                        unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                        unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                        // Load pixels
                        float2 pixel[NR_POLARIZATIONS];
                        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            unsigned int pixel_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                            pixel[pol] = subgrid[pixel_idx];
                        }

                        // Load first aterm
                        float2 *aterm1;
                        if (mode == UPDATE_MODE_DERIV) {
                            size_t station1_idx = index_aterm(subgrid_size, nr_terms, aterm_idx, term_offset+term_nr, y, x);
                            aterm1 = (float2 *) &aterm_derivatives[station1_idx];
                        } else {
                            size_t station1_idx = index_aterm(subgrid_size, nr_stations, aterm_idx, station1, y, x);
                            aterm1 = (float2 *) &aterm[station1_idx];
                        }

                        // Load second aterm
                        size_t station2_idx = index_aterm(subgrid_size, nr_stations, aterm_idx, station2, y, x);
                        float2 *aterm2 = (float2 *) &aterm[station2_idx];

                        // Apply aterm
                        apply_aterm_calibrate(pixel, aterm1, aterm2);

                        // Store pixels in shared memory
                        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            pixels_[term_nr][pol][x] = pixel[pol];
                        }
                    } // end if
                } // end for j (subgrid_size * terms)

                __syncthreads();

                // Iterate all columns of current row
                for (unsigned int x = 0; x < subgrid_size; x++) {

                    // Load l,m,n
                    float l = lmnp_[y][x].x;
                    float m = lmnp_[y][x].y;
                    float n = lmnp_[y][x].z;

                    // Load phase offset
                    float phase_offset = lmnp_[y][x].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                    // Iterate all terms
                    for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {

                            // Load pixels
                            float2 pixel_xx = pixels_[term_nr][0][x];
                            float2 pixel_xy = pixels_[term_nr][1][x];
                            float2 pixel_yx = pixels_[term_nr][2][x];
                            float2 pixel_yy = pixels_[term_nr][3][x];

                            // Update sums
                            sum_xx[term_nr].x += phasor.x * pixel_xx.x;
                            sum_xx[term_nr].y += phasor.x * pixel_xx.y;
                            sum_xx[term_nr].x -= phasor.y * pixel_xx.y;
                            sum_xx[term_nr].y += phasor.y * pixel_xx.x;

                            sum_xy[term_nr].x += phasor.x * pixel_xy.x;
                            sum_xy[term_nr].y += phasor.x * pixel_xy.y;
                            sum_xy[term_nr].x -= phasor.y * pixel_xy.y;
                            sum_xy[term_nr].y += phasor.y * pixel_xy.x;

                            sum_yx[term_nr].x += phasor.x * pixel_yx.x;
                            sum_yx[term_nr].y += phasor.x * pixel_yx.y;
                            sum_yx[term_nr].x -= phasor.y * pixel_yx.y;
                            sum_yx[term_nr].y += phasor.y * pixel_yx.x;

                            sum_yy[term_nr].x += phasor.x * pixel_yy.x;
                            sum_yy[term_nr].y += phasor.x * pixel_yy.y;
                            sum_yy[term_nr].x -= phasor.y * pixel_yy.y;
                            sum_yy[term_nr].y += phasor.y * pixel_yy.x;
                    } // end for term_nr
                } // end for x
            } // end for y

            // Scale sums and store in device memory
            for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
                const float scale = 1.0f / nr_pixels;
                if (time < nr_timesteps) {
                    unsigned int time_idx = time_offset_global + time;
                    unsigned int chan_idx = chan;
                    if (mode == UPDATE_MODE_DERIV) {
                        unsigned int sum_idx_xx = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr, 0, time_idx, chan_idx);
                        unsigned int sum_idx_xy = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr, 1, time_idx, chan_idx);
                        unsigned int sum_idx_yx = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr, 2, time_idx, chan_idx);
                        unsigned int sum_idx_yy = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr, 3, time_idx, chan_idx);
                        sum_deriv[sum_idx_xx] = sum_xx[term_nr] * scale;
                        sum_deriv[sum_idx_xy] = sum_xy[term_nr] * scale;
                        sum_deriv[sum_idx_yx] = sum_yx[term_nr] * scale;
                        sum_deriv[sum_idx_yy] = sum_yy[term_nr] * scale;
                    } else {
                        unsigned int sum_idx_xx = index_sum_aterm(total_nr_timesteps, nr_channels, 0, time_idx, chan_idx);
                        unsigned int sum_idx_xy = index_sum_aterm(total_nr_timesteps, nr_channels, 1, time_idx, chan_idx);
                        unsigned int sum_idx_yx = index_sum_aterm(total_nr_timesteps, nr_channels, 2, time_idx, chan_idx);
                        unsigned int sum_idx_yy = index_sum_aterm(total_nr_timesteps, nr_channels, 3, time_idx, chan_idx);
                        sum_aterm[sum_idx_xx] = sum_xx[term_nr] * scale;
                        sum_aterm[sum_idx_xy] = sum_xy[term_nr] * scale;
                        sum_aterm[sum_idx_yx] = sum_yx[term_nr] * scale;
                        sum_aterm[sum_idx_yy] = sum_yy[term_nr] * scale;
                    }
                }
            } // end for term_nr

            __syncthreads();

        } // end for i (visibilities)
    } // end for time_offset_local
} // end update_sums


__device__ void update_gradient(
    const int                         subgrid_size,
    const float                       image_size,
    const unsigned int                total_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_stations,
    const unsigned int                nr_terms,
    const UVW*           __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sum_aterm,
          float2*        __restrict__ sum_deriv,
          float2*        __restrict__ gradient,
          float4                      lmnp_[MAX_SUBGRID_SIZE][MAX_SUBGRID_SIZE])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;

    // Load metadata for first subgrid
    const Metadata &m0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset  = m.time_index - m0.time_index;
    const unsigned int nr_timesteps = m.nr_timesteps;

    // Shared memory
    __shared__ float2 residual_[NR_POLARIZATIONS][MAX_NR_THREADS];
    __shared__ float2 gradient_[MAX_NR_TERMS];

    // Reset shared memory
    for (unsigned int i = tid; i < MAX_NR_TERMS; i += nr_threads) {
        if (i < MAX_NR_TERMS) {
            gradient_[i] = make_float2(0, 0);
        }
    }

    // Iterate all visibilities
    for (unsigned int i = tid; i < ALIGN(nr_timesteps*nr_channels, nr_threads); i += nr_threads) {
        unsigned int time = i / nr_channels;
        unsigned int chan = i % nr_channels;

        // Compute residual visibilities
        if (time < nr_timesteps) {
            unsigned int time_idx = time_offset + time;
            unsigned int chan_idx = chan;
            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                unsigned int vis_idx = index_visibility(nr_channels, time_idx, chan_idx, pol);
                unsigned int sum_idx = index_sum_aterm(total_nr_timesteps, nr_channels, pol, time_idx, chan_idx);
                residual_[pol][tid] = (visibilities[vis_idx] - sum_aterm[sum_idx]) * weights[vis_idx];
            }
        } else {
            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                residual_[pol][tid] = make_float2(0, 0);
            }
        }

        __syncthreads();

        // Iterate all terms
        for (unsigned term_nr = tid; term_nr < nr_terms; term_nr += nr_threads) {

            // Compute gradient update
            float2 update = make_float2(0, 0);

            // Iterate current batch of residuals
            for (unsigned j = 0; j < MAX_NR_THREADS; j++) {
                // Compute the index of the first visibility in the batch
                unsigned int k = i - tid + j;

                // Derive the current time and channel
                unsigned int time = k / nr_channels;
                unsigned int chan = k % nr_channels;

                if (term_nr < nr_terms && time < nr_timesteps) {
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned int time_idx = time_offset + time;
                        unsigned int chan_idx = chan;
                        unsigned int sum_idx = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr, pol, time_idx, chan_idx);
                        float2 sum      = sum_deriv[sum_idx];
                        float2 residual = residual_[pol][j];

                        if (term_nr < nr_terms) {
                            update.x += sum.x * residual.x;
                            update.x += sum.y * residual.y;
                            update.y += sum.x * residual.y;
                            update.y -= sum.y * residual.x;
                        }
                    } // end for pol
                } // end if
            } // end for threads

            // Update local gradient
            gradient_[term_nr] += update;
        } // end for term_nr

        __syncthreads();

    } // end for i (visibilities)

    // Iterate all terms * terms
    for (unsigned int i = tid; i < nr_terms; i += nr_threads) {
        if (i < nr_terms) {
            atomicAdd(&gradient[i], gradient_[i]);
        }
    }
} // end update_gradient


__device__ void update_hessian(
    const unsigned int                total_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_stations,
    const unsigned int                nr_terms,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ sum_aterm,
          float2*        __restrict__ sum_deriv,
          float2*        __restrict__ hessian)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned s          = blockIdx.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    // Metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset  = m.time_index;
    const unsigned int nr_timesteps = m.nr_timesteps;

    // Iterate all terms * terms
    for (unsigned int term_nr = tid; term_nr < (nr_terms*nr_terms); term_nr += nr_threads) {
        unsigned term_nr0 = term_nr / nr_terms;
        unsigned term_nr1 = term_nr % nr_terms;

        // Compute hessian update
        float2 update = make_float2(0, 0);

        // Iterate all timesteps
        for (unsigned int time = 0; time < nr_timesteps; time++) {

            // Iterate all channels
            for (unsigned int chan = 0; chan < nr_channels; chan++) {

                // Iterate all polarizations
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    unsigned int time_idx = time_offset + time;
                    unsigned int chan_idx = chan;
                    unsigned int  vis_idx = index_visibility(nr_channels, time_idx, chan_idx, pol);
                    unsigned int sum_idx0 = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr0, pol, time_idx, chan_idx);
                    unsigned int sum_idx1 = index_sum_deriv(total_nr_timesteps, nr_channels, term_nr1, pol, time_idx, chan_idx);
                    float2 sum0 = sum_deriv[sum_idx0];
                    float2 sum1 = sum_deriv[sum_idx1] * weights[vis_idx];

                    // Update hessian
                    if (term_nr0 < nr_terms) {
                        update.x += sum0.x * sum1.x;
                        update.x += sum0.y * sum1.y;
                        update.y += sum0.y * sum1.x;
                        update.y -= sum0.x * sum1.y;
                    }
                } // end for pol
            } // end chan
        } // end for time

        // Update local hessian
        if (term_nr0 < nr_terms) {
            atomicAdd(&hessian[term_nr], update);
        }
    } // end for term_nr (terms * terms)
} // end update_hessian


#define UPDATE_SUMS(current_nr_terms, mode) \
    for (; (term_offset + current_nr_terms) <= nr_terms; term_offset += current_nr_terms) { \
        update_sums<current_nr_terms, mode>( \
                subgrid_size, image_size, total_nr_timesteps, nr_channels, nr_stations, \
                nr_terms, term_offset, \
                uvw, aterm, aterm_indices, aterm_derivatives, wavenumbers, metadata, subgrid, \
                sum_aterm, sum_deriv, lmnp_, pixels_); \
    }

extern "C" {

__global__ void kernel_calibrate_sums(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         total_nr_timesteps,
    const int                         nr_channels,
    const int                         nr_stations,
    const int                         nr_terms,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const int*           __restrict__ aterm_indices,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sum_aterm,
          float2*        __restrict__ sum_deriv,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    // Shared memory
    __shared__ float4 lmnp_[MAX_SUBGRID_SIZE][MAX_SUBGRID_SIZE];
    __shared__ float2 pixels_[MAX_NR_TERMS][NR_POLARIZATIONS][MAX_SUBGRID_SIZE];

    compute_lmnp(grid_size, subgrid_size, image_size, w_step, metadata, lmnp_);

    int term_offset;

    // Compute sum_deriv
    term_offset = 0;
    UPDATE_SUMS(8, UPDATE_MODE_DERIV)
    UPDATE_SUMS(7, UPDATE_MODE_DERIV)
    UPDATE_SUMS(6, UPDATE_MODE_DERIV)
    UPDATE_SUMS(5, UPDATE_MODE_DERIV)
    UPDATE_SUMS(4, UPDATE_MODE_DERIV)
    UPDATE_SUMS(3, UPDATE_MODE_DERIV)
    UPDATE_SUMS(2, UPDATE_MODE_DERIV)
    UPDATE_SUMS(1, UPDATE_MODE_DERIV)

    // Compute sum_aterm
    term_offset = 0;
    UPDATE_SUMS(1, UPDATE_MODE_ATERM)
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
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const int*           __restrict__ aterm_indices,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sum_aterm,
          float2*        __restrict__ sum_deriv,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    __shared__ float4 lmnp_[MAX_SUBGRID_SIZE][MAX_SUBGRID_SIZE];

    compute_lmnp(grid_size, subgrid_size, image_size, w_step, metadata, lmnp_);

    update_gradient(
        subgrid_size, image_size, total_nr_timesteps,
        nr_channels, nr_stations, nr_terms,
        uvw, aterm, aterm_derivatives,
        wavenumbers, visibilities, weights, metadata, subgrid,
        sum_aterm, sum_deriv, gradient, lmnp_);
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
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float*         __restrict__ weights,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const int*           __restrict__ aterm_indices,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ sum_aterm,
          float2*        __restrict__ sum_deriv,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    update_hessian(
        total_nr_timesteps, nr_channels, nr_stations, nr_terms,
        visibilities, weights, metadata, sum_aterm, sum_deriv, hessian);
} // end kernel_calibrate_hessian

} // end extern "C"
