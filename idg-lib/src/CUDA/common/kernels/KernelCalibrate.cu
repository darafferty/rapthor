#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS     8
#define MAX_SUBGRID_SIZE 32
#define MAX_NR_THREADS   128
#define MAX_NR_TIMESTEPS 128

// Shared memory
__shared__ float4 lmnp_[MAX_SUBGRID_SIZE][MAX_SUBGRID_SIZE];
__shared__ float2 pixels_[MAX_NR_TERMS][NR_POLARIZATIONS][MAX_SUBGRID_SIZE];
__shared__ float2 gradient_[MAX_NR_TERMS];
__shared__ float2 residual_[NR_POLARIZATIONS][MAX_NR_THREADS];


// Index in scratch_sum
inline __device__ long index_sums(
    unsigned int max_nr_timesteps,
    unsigned int nr_channels,
    unsigned int s,
    unsigned int time,
    unsigned int chan,
    unsigned int term_nr,
    unsigned int pol)
{
    // sums: [nr_subgrids][MAX_NR_TERMS][NR_POLARIZATIONS][max_nr_timesteps][nr_channels]
    return s * MAX_NR_TERMS * NR_POLARIZATIONS * max_nr_timesteps * nr_channels +
           term_nr * NR_POLARIZATIONS * max_nr_timesteps * nr_channels +
           pol * max_nr_timesteps * nr_channels +
           time * nr_channels +
           chan;
}


__device__ void compute_lmnp(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const Metadata*      __restrict__ metadata)
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


template<int current_nr_terms>
__device__ void update_sums(
    const int                         subgrid_size,
    const float                       image_size,
    const unsigned int                max_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_terms,
    const unsigned int                term_offset,
    const UVW*           __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ scratch_sum)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Metadata for first subgrid
    const Metadata &m_0       = metadata[0];

    // metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset  = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const unsigned int station2     = m.baseline.station2;
    const unsigned int nr_timesteps = m.nr_timesteps;

    // Iterate all visibilities
    for (unsigned int i = tid; i < ALIGN(nr_timesteps*nr_channels, nr_threads); i += nr_threads) {
        unsigned int time = i / nr_channels;
        unsigned int chan = i % nr_channels;

        // Load UVW
        float u, v, w;
        if (time < nr_timesteps) {
            u = uvw[time_offset + time].u;
            v = uvw[time_offset + time].v;
            w = uvw[time_offset + time].w;
        }

        // Load wavenumber
        float wavenumber = wavenumbers[chan];

        // Accumulate sums in registers
        float2 sum[current_nr_terms][NR_POLARIZATIONS];
        for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                sum[term_nr][pol] = make_float2(0, 0);
            }
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

                    // Load aterm derivative
                    size_t station1_idx = index_aterm(subgrid_size, 0, 0, term_offset+term_nr, y, x);
                    float2 *aterm1 = (float2 *) &aterm_derivatives[station1_idx];

                    // Load second aterm
                    size_t station2_idx = index_aterm(subgrid_size, 0, 0, station2, y, x);
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

                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        // Load pixel
                        float2 pixel = pixels_[term_nr][pol][x];

                        // Update sums
                        sum[term_nr][pol].x += phasor.x * pixel.x;
                        sum[term_nr][pol].y += phasor.x * pixel.y;
                        sum[term_nr][pol].x -= phasor.y * pixel.y;
                        sum[term_nr][pol].y += phasor.y * pixel.x;
                    }
                } // end for term_nr
            } // end for x
        } // end for y

        // Scale sums and store in device memory
        for (unsigned int term_nr = 0; term_nr < current_nr_terms; term_nr++) {
            const float scale = 1.0f / nr_pixels;
            if (time < nr_timesteps) {
                for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    unsigned int sum_idx = index_sums(max_nr_timesteps, nr_channels, s, time, chan, term_offset+term_nr, pol);
                    scratch_sum[sum_idx] = sum[term_nr][pol] * scale;
                }
            }
        } // end for term_nr

        __syncthreads();

    } // end for i (visibilities)
} // end update_sums


__device__ void update_gradient(
    const int                         subgrid_size,
    const float                       image_size,
    const unsigned int                max_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_terms,
    const UVW*           __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ scratch_sum,
          float2*        __restrict__ gradient)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned s          = blockIdx.x;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Metadata for first subgrid
    const Metadata &m_0       = metadata[0];

    // metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset  = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const unsigned int station1     = m.baseline.station1;
    const unsigned int station2     = m.baseline.station2;
    const unsigned int nr_timesteps = m.nr_timesteps;

    // Reset shared memory
    for (unsigned int i = tid; i < MAX_NR_TERMS; i += nr_threads) {
        if (i < MAX_NR_TERMS) {
            gradient_[i] = make_float2(0, 0);
        }
    }

    // Iterate all visibilities
    for (unsigned int i = tid; i < ALIGN(nr_timesteps*nr_channels, nr_threads); i+= nr_threads) {
        unsigned int time = i / nr_channels;
        unsigned int chan = i % nr_channels;

        // Load UVW
        float u, v, w;
        if (time < nr_timesteps) {
            u = uvw[time_offset + time].u;
            v = uvw[time_offset + time].v;
            w = uvw[time_offset + time].w;
        }

        // Load wavenumber
        float wavenumber = wavenumbers[chan];

        // Accumulate sums in registers
        float2 sum[NR_POLARIZATIONS];
        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
            sum[pol] = make_float2(0, 0);
        }

        // Iterate all rows of the subgrid
        for (unsigned int y = 0; y < subgrid_size; y++) {
            __syncthreads();

            // Precompute data for one row
            for (unsigned x = tid; x < subgrid_size; x += nr_threads) {

                if (x < subgrid_size) {
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
                    size_t station1_idx = index_aterm(subgrid_size, 0, 0, station1, y, x);
                    float2 *aterm1 = (float2 *) &aterm[station1_idx];

                    // Load second aterm
                    size_t station2_idx = index_aterm(subgrid_size, 0, 0, station2, y, x);
                    float2 *aterm2 = (float2 *) &aterm[station2_idx];

                    // Apply aterm
                    apply_aterm_calibrate(pixel, aterm1, aterm2);

                    // Store pixels in shared memory
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixels_[0][pol][x] = pixel[pol];
                    }
                } // end if
            } // end for x

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

                // Update sums
                for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    sum[pol].x += phasor.x * pixels_[0][pol][x].x;
                    sum[pol].y += phasor.x * pixels_[0][pol][x].y;
                    sum[pol].x -= phasor.y * pixels_[0][pol][x].y;
                    sum[pol].y += phasor.y * pixels_[0][pol][x].x;
                }
            } // end for x
        } // end for y

        // Scale sums and store in shared memory
        const float scale = 1.0f / nr_pixels;
        if (time < nr_timesteps) {
            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                unsigned int time_idx = time_offset + time;
                unsigned int chan_idx = chan;
                unsigned int vis_idx  = index_visibility(nr_channels, time_idx, chan_idx, pol);
                residual_[pol][tid] = visibilities[vis_idx] - sum[pol] * scale;
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

            for (unsigned j = 0; j < MAX_NR_THREADS; j++) {
                unsigned int k = i - tid + j;
                unsigned int time_ = k / nr_channels;
                unsigned int chan_ = k % nr_channels;

                if (time < nr_timesteps) {
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned int sum_idx = index_sums(max_nr_timesteps, nr_channels, s, time_, chan_, term_nr, pol);
                        float2 sum      = scratch_sum[sum_idx];
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
    const unsigned int                max_nr_timesteps,
    const unsigned int                nr_channels,
    const unsigned int                nr_terms,
    const float2*        __restrict__ visibilities,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ scratch_sum,
          float2*        __restrict__ hessian)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned s          = blockIdx.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    // metadata for current subgrid
    const Metadata &m = metadata[s];
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
                    unsigned int sum_idx0 = index_sums(max_nr_timesteps, nr_channels, s, time, chan, term_nr0, pol);
                    unsigned int sum_idx1 = index_sums(max_nr_timesteps, nr_channels, s, time, chan, term_nr1, pol);
                    float2 sum0 = scratch_sum[sum_idx1];
                    float2 sum1 = scratch_sum[sum_idx0];

                    // Update hessian
                    if (term_nr0 < nr_terms) {
                        update.x += sum0.x * sum1.x;
                        update.x += sum0.y * sum1.y;
                        update.y += sum0.x * sum1.y;
                        update.y -= sum0.y * sum1.x;
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


#define UPDATE_SUMS(current_nr_terms) \
    for (; (term_offset + current_nr_terms) <= nr_terms; term_offset += current_nr_terms) { \
        update_sums<current_nr_terms>( \
                subgrid_size, image_size, max_nr_timesteps, nr_channels, \
                nr_terms, term_offset, \
                uvw, aterm, aterm_derivatives, wavenumbers, metadata, \
                subgrid, scratch_sum); \
    }

extern "C" {

__global__ void kernel_calibrate(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         max_nr_timesteps,
    const int                         nr_channels,
    const int                         nr_terms,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
    const float2*        __restrict__ visibilities,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ scratch_sum,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    compute_lmnp(grid_size, subgrid_size, image_size, w_step, metadata);

    int term_offset = 0;
    UPDATE_SUMS(8)
    UPDATE_SUMS(7)
    UPDATE_SUMS(6)
    UPDATE_SUMS(5)
    UPDATE_SUMS(4)
    UPDATE_SUMS(3)
    UPDATE_SUMS(2)
    UPDATE_SUMS(1)

    update_gradient(
        subgrid_size, image_size, max_nr_timesteps,
        nr_channels, nr_terms,
        uvw, aterm, aterm_derivatives,
        wavenumbers, visibilities, metadata, subgrid, scratch_sum, gradient);

    update_hessian(
        max_nr_timesteps, nr_channels, nr_terms,
        visibilities, metadata, scratch_sum, hessian);
} // end kernel_calibrate

} // end extern "C"
