#include "Types.h"
#include "math.cu"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define MAX_NR_TERMS     8
#define MAX_SUBGRID_SIZE 32
#define MAX_NR_THREADS   128

// Index in scratch_sum
inline __device__ long index_sums(
    unsigned int s,
    unsigned int tid,
    unsigned int term_nr,
    unsigned int pol)
{
    // sums: [nr_subgrids][MAX_NR_THREADS][MAX_NR_TERMS][NR_POLARIZATIONS]
    return s * MAX_NR_THREADS * MAX_NR_TERMS * NR_POLARIZATIONS +
           tid * MAX_NR_TERMS * NR_POLARIZATIONS +
           term_nr * NR_POLARIZATIONS +
           pol;
}


__device__ void initialize_shared_memory(
    float2 pixels_[NR_POLARIZATIONS][MAX_SUBGRID_SIZE][MAX_NR_TERMS],
    float2 sums_[NR_POLARIZATIONS][MAX_NR_TERMS],
    float2 hessian_[MAX_NR_TERMS][MAX_NR_TERMS],
    float2 gradient_[MAX_NR_TERMS])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    for (unsigned int i = tid; i < (NR_POLARIZATIONS*MAX_SUBGRID_SIZE*MAX_NR_TERMS); i += nr_threads) {
        pixels_[0][0][i] = make_float2(0, 0);
    }

    for (unsigned int i = tid; i < (MAX_NR_TERMS*MAX_NR_TERMS); i += nr_threads) {
        if (i < MAX_NR_TERMS) {
            gradient_[i] = make_float2(0, 0);
        }

        if (i < (MAX_NR_TERMS*MAX_NR_TERMS)) {
            hessian_[0][i] = make_float2(0, 0);
        }
    }

    for (unsigned int i = tid; i < (NR_POLARIZATIONS*MAX_NR_TERMS); i += nr_threads) {
        sums_[0][i] = make_float2(0, 0);
    }

    __syncthreads();
} // end initialize_shared_memory


template<int nr_terms>
__device__ void update_sums(
    const int                         subgrid_size,
    const float                       image_size,
    const unsigned int                s,
    const unsigned int                time,
    const unsigned int                chan,
    const UVW                         uvw_offset,
    const UVW*           __restrict__ uvw,
    const float2*        __restrict__ aterm,
    const float2*        __restrict__ aterm_derivatives,
    const float*         __restrict__ wavenumbers,
    const Metadata*      __restrict__ metadata,
    const float2*        __restrict__ subgrid,
          float2*        __restrict__ scratch_sum,
          float2 pixels_[NR_POLARIZATIONS][MAX_SUBGRID_SIZE][MAX_NR_TERMS],
          float4 lmn_[MAX_SUBGRID_SIZE])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Metadata for first subgrid
    const Metadata &m_0       = metadata[0];

    // metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset  = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const unsigned int station1     = m.baseline.station1;
    const unsigned int station2     = m.baseline.station2;
    const unsigned int nr_timesteps = m.nr_timesteps;

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
    float2 sumXX[MAX_NR_TERMS];
    float2 sumXY[MAX_NR_TERMS];
    float2 sumYX[MAX_NR_TERMS];
    float2 sumYY[MAX_NR_TERMS];
    for (unsigned int term_nr = 0; term_nr < (nr_terms+1); term_nr++) {
        sumXX[term_nr] = make_float2(0, 0);
        sumXY[term_nr] = make_float2(0, 0);
        sumYX[term_nr] = make_float2(0, 0);
        sumYY[term_nr] = make_float2(0, 0);
    }

    // Iterate all rows of the subgrid
    for (unsigned int y = 0; y < subgrid_size; y++) {
        __syncthreads();

        // Precompute data for one row
        for (unsigned x = tid; x < subgrid_size; x += nr_threads) {

            if (x < subgrid_size) {
                // Precompute l,m,n and phase offset
                float l = compute_l(x, subgrid_size, image_size);
                float m = compute_m(y, subgrid_size, image_size);
                float n = compute_n(l, m);
                float phase_offset = uvw_offset.u*l + uvw_offset.v*m + uvw_offset.w*n;
                lmn_[x] = make_float4(l, m, n, phase_offset);

                // Precompute pixels
                for (unsigned term_nr = 0; term_nr < (nr_terms+1); term_nr++) {
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

                    if (term_nr == nr_terms) {
                        // Load aterm for station1
                        size_t station1_idx = index_aterm(subgrid_size, 0, 0, station1, y, x);
                        aterm1 = (float2 *) &aterm[station1_idx];
                    } else {
                        // Load aterm derivative
                        size_t station1_idx = index_aterm(subgrid_size, 0, 0, term_nr, y, x);
                        aterm1 = (float2 *) &aterm_derivatives[station1_idx];
                    }

                    // Load second aterm
                    size_t station2_idx = index_aterm(subgrid_size, 0, 0, station2, y, x);
                    float2 *aterm2 = (float2 *) &aterm[station2_idx];

                    // Apply aterm
                    apply_aterm_calibrate(pixel, aterm1, aterm2);

                    // Store pixels in shared memory
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixels_[pol][x][term_nr] = pixel[pol];
                    }
                } // end for terms
            } // end if
        } // end for x

        __syncthreads();

        // Iterate all columns of current row
        for (unsigned int x = 0; x < subgrid_size; x++) {

            // Load l,m,n
            float l = lmn_[x].x;
            float m = lmn_[x].y;
            float n = lmn_[x].z;

            // Load phase offset
            float phase_offset = lmn_[x].w;

            // Compute phase index
            float phase_index = u*l + v*m + w*n;

            // Compute phasor
            float  phase  = (phase_index * wavenumber) - phase_offset;
            float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

            // Iterate all terms
            for (unsigned int term_nr = 0; term_nr < MAX_NR_TERMS; term_nr++) {

                // Load pixels
                float2 pixelXX = pixels_[0][x][term_nr];
                float2 pixelXY = pixels_[1][x][term_nr];
                float2 pixelYX = pixels_[2][x][term_nr];
                float2 pixelYY = pixels_[3][x][term_nr];

                // Update sums
                sumXX[term_nr].x += phasor.x * pixelXX.x;
                sumXX[term_nr].y += phasor.x * pixelXX.y;
                sumXX[term_nr].x -= phasor.y * pixelXX.y;
                sumXX[term_nr].y += phasor.y * pixelXX.x;

                sumXY[term_nr].x += phasor.x * pixelXY.x;
                sumXY[term_nr].y += phasor.x * pixelXY.y;
                sumXY[term_nr].x -= phasor.y * pixelXY.y;
                sumXY[term_nr].y += phasor.y * pixelXY.x;

                sumYX[term_nr].x += phasor.x * pixelYX.x;
                sumYX[term_nr].y += phasor.x * pixelYX.y;
                sumYX[term_nr].x -= phasor.y * pixelYX.y;
                sumYX[term_nr].y += phasor.y * pixelYX.x;

                sumYY[term_nr].x += phasor.x * pixelYY.x;
                sumYY[term_nr].y += phasor.x * pixelYY.y;
                sumYY[term_nr].x -= phasor.y * pixelYY.y;
                sumYY[term_nr].y += phasor.y * pixelYY.x;
            } // end for term_nr
        } // end for x
    } // end for y

    // Scale sums and store in device memory
    for (unsigned int term_nr = 0; term_nr < MAX_NR_TERMS; term_nr++) {
        const float scale = 1.0f / nr_pixels;
        if (time < nr_timesteps) {
            unsigned int sum_idx = index_sums(s, tid, term_nr, 0);
            float4 *sum_ptr = (float4 *) &scratch_sum[sum_idx];
            float4 sumA = make_float4(sumXX[term_nr].x, sumXX[term_nr].y, sumXY[term_nr].x, sumYX[term_nr].y);
            float4 sumB = make_float4(sumYX[term_nr].x, sumYX[term_nr].y, sumYY[term_nr].x, sumYY[term_nr].y);
            sum_ptr[0] = sumA * scale;
            sum_ptr[1] = sumB * scale;
        }
    } // end for term_nr

    __syncthreads();
} // end update_sums


template<int nr_terms>
__device__ void update_local_solution(
    const unsigned int                s,
    const unsigned                    visibility_offset,
    const unsigned int                nr_channels,
    const float2*        __restrict__ visibilities,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ scratch_sum,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient,
          float2 sums_[NR_POLARIZATIONS][MAX_NR_TERMS],
          float2 gradient_[MAX_NR_TERMS],
          float2 hessian_[MAX_NR_TERMS][MAX_NR_TERMS])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    // Metadata for first subgrid
    const Metadata &m_0       = metadata[0];

    // metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset  = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const unsigned int nr_timesteps = m.nr_timesteps;

    // Iterate all visibilities
    for (unsigned int j = 0; j < MAX_NR_THREADS; j++) {
        unsigned int k = (visibility_offset - tid) + j;
        unsigned int time = k / nr_channels;
        unsigned int chan = k % nr_channels;

        if (time < nr_timesteps) {
            // Load sums for current visibility
            for (unsigned int term_nr = tid; term_nr < MAX_NR_TERMS; term_nr += nr_threads) {
                unsigned int sum_idx = index_sums(s, j, term_nr, 0);
                float4 *sum_ptr = (float4 *) &scratch_sum[sum_idx];
                float4 a = sum_ptr[0];
                float4 b = sum_ptr[1];
                sums_[0][term_nr] = make_float2(a.x, a.y);
                sums_[1][term_nr] = make_float2(a.z, a.w);
                sums_[2][term_nr] = make_float2(b.x, b.y);
                sums_[3][term_nr] = make_float2(b.z, b.w);
            }

            __syncthreads();

            // Compute residual visibility
            float2 visibility_res[NR_POLARIZATIONS];
            for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                unsigned int time_idx = time_offset + time;
                unsigned int chan_idx = chan;
                unsigned int vis_idx  = index_visibility(nr_channels, time_idx, chan_idx, pol);
                visibility_res[pol] = visibilities[vis_idx] - sums_[pol][nr_terms];
            }

            // Iterate all terms * terms
            for (unsigned int term_nr = tid; term_nr < (nr_terms*nr_terms); term_nr += nr_threads) {
                unsigned term_nr0 = term_nr / nr_terms;
                unsigned term_nr1 = term_nr % nr_terms;

                // Iterate all polarizations
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {

                    // Update local gradient
                    if (term_nr < nr_terms) {
                        gradient_[term_nr].x +=
                            sums_[pol][term_nr].x * visibility_res[pol].x +
                            sums_[pol][term_nr].y * visibility_res[pol].y;
                        gradient_[term_nr].y +=
                            sums_[pol][term_nr].x * visibility_res[pol].y -
                            sums_[pol][term_nr].y * visibility_res[pol].x;
                    }

                    // Update local hessian
                    if (term_nr < (nr_terms*nr_terms)) {
                        hessian_[term_nr1][term_nr0].x +=
                            sums_[pol][term_nr0].x * sums_[pol][term_nr1].x +
                            sums_[pol][term_nr0].y * sums_[pol][term_nr1].y;
                        hessian_[term_nr0][term_nr1].y +=
                            sums_[pol][term_nr0].x * sums_[pol][term_nr1].y -
                            sums_[pol][term_nr0].y * sums_[pol][term_nr1].x;
                    }
                } // end for pol
            } // end for term_nr (terms * terms)
        } // end if time
    } // end for j (nr_threads)
} // end update_local_solution


__device__ void update_global_solution(
    const unsigned int nr_terms,
    float2*        __restrict__ hessian,
    float2*        __restrict__ gradient,
    float2                      hessian_[MAX_NR_TERMS][MAX_NR_TERMS],
    float2                      gradient_[MAX_NR_TERMS])
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    // Iterate all terms * terms
    for (unsigned int i = tid; i < (nr_terms*nr_terms); i += nr_threads) {
        unsigned term_nr0 = i / nr_terms;
        unsigned term_nr1 = i % nr_terms;

        if (i < nr_terms) {
            atomicAdd(&gradient[i], gradient_[i]);
        }

        if (i < (nr_terms*nr_terms)) {
            atomicAdd(&hessian[i], hessian_[term_nr1][term_nr0]);
        }
    } // end for i
} // end update_global_solution


extern "C" {

__global__ void kernel_calibrate(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
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
    unsigned s          = blockIdx.x;
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int nr_timesteps      = m.nr_timesteps;
    const int x_coordinate      = m.coordinate.x;
    const int y_coordinate      = m.coordinate.y;
    const int z_coordinate      = m.coordinate.z;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) z_coordinate + 0.5) * 2 * M_PI;
    const UVW uvw_offset = (UVW) { u_offset, v_offset, w_offset };

    __shared__ float4 lmn_[MAX_SUBGRID_SIZE];
    __shared__ float2 pixels_[NR_POLARIZATIONS][MAX_SUBGRID_SIZE][MAX_NR_TERMS];
    __shared__ float2 sums_[NR_POLARIZATIONS][MAX_NR_TERMS];
    __shared__ float2 hessian_[MAX_NR_TERMS][MAX_NR_TERMS];
    __shared__ float2 gradient_[MAX_NR_TERMS];

    /*
        Phase 0: initialize shared memory to zero
    */
    initialize_shared_memory(pixels_, sums_, hessian_, gradient_);

    // Iterate all visibilities
    for (unsigned int visibility_offset = tid; visibility_offset < ALIGN(nr_timesteps*nr_channels, nr_threads); visibility_offset += nr_threads) {
        unsigned int time = visibility_offset / nr_channels;
        unsigned int chan = visibility_offset % nr_channels;

        /*
            Phase 1: "degrid" all subgrids, row by row
        */
        update_sums<6>(
                subgrid_size, image_size, s, time, chan,
                uvw_offset, uvw, aterm, aterm_derivatives, wavenumbers, metadata,
                subgrid, scratch_sum, pixels_, lmn_);

        /*
            Phase 2: update local gradient and hessian
        */
        update_local_solution<6>(
            s, visibility_offset, nr_channels,
            visibilities, metadata, scratch_sum,
            hessian, gradient, sums_, gradient_, hessian_);
    } // end for visibility_offset

    __syncthreads();

    /*
        Phase 3: update global gradient and hessian
    */
    update_global_solution(nr_terms, hessian, gradient, hessian_, gradient_);
} // end kernel_calibrate

} // end extern "C"
