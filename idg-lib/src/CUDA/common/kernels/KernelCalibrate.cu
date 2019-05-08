#include "Types.h"
#include "math.cu"

#define MAX_NR_TERMS 8

// Index in scratch_sum
inline __device__ long index_sums(
    unsigned int nr_timesteps,
    unsigned int nr_channels,
    unsigned int nr_terms,
    unsigned int s,
    unsigned int time,
    unsigned int chan,
    unsigned int pol,
    unsigned int term_nr)
{
    // sums: [nr_subgrids][nr_timesteps][nr_channels][nr_terms][NR_POLARIZATIONS]
    return s * nr_timesteps * nr_channels * nr_terms * NR_POLARIZATIONS +
           time * nr_channels * nr_terms * NR_POLARIZATIONS +
           chan * nr_terms * NR_POLARIZATIONS +
           term_nr * NR_POLARIZATIONS +
           pol;
}

// Index in scratch_pix
inline __device__ long index_pixels(
    unsigned int subgrid_size,
    unsigned int s,
    unsigned int term_nr,
    unsigned int pol,
    unsigned int y,
    unsigned int x)
{
    // pix: [nr_subgrids][MAX_NR_TERMS][subgrid_size][subgrid_size][NR_POLARIZATIONS]
    return s * MAX_NR_TERMS * subgrid_size * subgrid_size * NR_POLARIZATIONS +
           term_nr * subgrid_size * subgrid_size * NR_POLARIZATIONS +
           y * subgrid_size * NR_POLARIZATIONS +
           x * NR_POLARIZATIONS +
           pol;
}

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
          float2*        __restrict__ scratch_pix,
          float2*        __restrict__ scratch_sum,
          float2*        __restrict__ hessian,
          float2*        __restrict__ gradient)
{
    unsigned s          = blockIdx.x;
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned tid        = tidx + tidy * blockDim.x;
    unsigned nr_threads = blockDim.x * blockDim.y;
    unsigned nr_pixels  = subgrid_size * subgrid_size;

    // Find offset of first subgrid
    const Metadata &m_0       = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const unsigned int station1 = m.baseline.station1;
    const unsigned int station2 = m.baseline.station2;
    const int nr_timesteps      = m.nr_timesteps;
    const int x_coordinate      = m.coordinate.x;
    const int y_coordinate      = m.coordinate.y;
    const int z_coordinate      = m.coordinate.z;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) z_coordinate + 0.5) * 2 * M_PI;

    /*
        Phase 1: apply aterm to subgrids and store prepared subgrid in device memory
    */

    // Apply aterm to subgrid
    for (unsigned i = tid; i < nr_pixels; i += nr_threads) {
        if (i < nr_pixels) {
            for (unsigned term_nr = 0; term_nr < (nr_terms+1); term_nr++) {
                unsigned int y = i / subgrid_size;
                unsigned int x = i % subgrid_size;

                // Compute shifted position in subgrid
                unsigned int x_src = (x + (subgrid_size/2)) % subgrid_size;
                unsigned int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Load pixels
                unsigned int pixel_idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
                unsigned int pixel_idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
                unsigned int pixel_idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
                unsigned int pixel_idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
                float2 pixelXX = subgrid[pixel_idx_xx];
                float2 pixelXY = subgrid[pixel_idx_xy];
                float2 pixelYX = subgrid[pixel_idx_yx];
                float2 pixelYY = subgrid[pixel_idx_yy];

                // Load first aterm
                float2 aXX1, aXY1, aYX1, aYY1;

                if (term_nr == nr_terms) {
                    // Load aterm for station1
                    size_t station1_idx = index_aterm(subgrid_size, 0, 0, station1, y, x);
                    aXX1 = aterm[station1_idx + 0];
                    aXY1 = aterm[station1_idx + 1];
                    aYX1 = aterm[station1_idx + 2];
                    aYY1 = aterm[station1_idx + 3];
                } else {
                    // Load aterm derivative
                    size_t station1_idx = index_aterm(subgrid_size, 0, 0, term_nr, y, x);
                    aXX1 = aterm_derivatives[station1_idx + 0];
                    aXY1 = aterm_derivatives[station1_idx + 1];
                    aYX1 = aterm_derivatives[station1_idx + 2];
                    aYY1 = aterm_derivatives[station1_idx + 3];
                }

                // Load second aterm
                float2 aXX2, aXY2, aYX2, aYY2;
                size_t station2_idx = index_aterm(subgrid_size, 0, 0, station2, y, x);
                aXX2 = aterm[station2_idx + 0];
                aXY2 = aterm[station2_idx + 1];
                aYX2 = aterm[station2_idx + 2];
                aYY2 = aterm[station2_idx + 3];

                // Apply aterm
                apply_aterm(
                    aXX1, aYX1, aXY1, aYY1,
                    aXX2, aYX2, aXY2, aYY2,
                    pixelXX, pixelXY, pixelYX, pixelYY);

                // Store pixels
                unsigned int pixel_idx = index_pixels(subgrid_size, s, term_nr, 0, y, x);
                scratch_pix[pixel_idx + 0] = pixelXX;
                scratch_pix[pixel_idx + 1] = pixelXY;
                scratch_pix[pixel_idx + 2] = pixelYX;
                scratch_pix[pixel_idx + 3] = pixelYY;
            } // end for terms
        } // end if
    } // end for pixels

    __syncthreads();

    __shared__ float2 sums_[NR_POLARIZATIONS][MAX_NR_TERMS];
    __shared__ float2 gradient_[MAX_NR_TERMS];
    __shared__ float2 hessian_[MAX_NR_TERMS][MAX_NR_TERMS];

    // Initialize shared memory to zero
    for (unsigned int i = tid; i < (MAX_NR_TERMS*MAX_NR_TERMS); i += nr_threads) {

        if (i < MAX_NR_TERMS) {
            gradient_[i] = make_float2(0, 0);
            sums_[0][i]  = make_float2(0, 0);
            sums_[1][i]  = make_float2(0, 0);
            sums_[2][i]  = make_float2(0, 0);
            sums_[3][i]  = make_float2(0, 0);
        }

        if (i < (MAX_NR_TERMS*MAX_NR_TERMS)) {
            hessian_[0][i] = make_float2(0, 0);
        }

    } // end for i

    __syncthreads();

    /*
        Phase 2: "degrid" all prepared subgrids, store results in local memory
    */

    // Iterate all timesteps
    for (unsigned int i = tid; i < (nr_timesteps*nr_channels); i += nr_threads) {
        unsigned int time = i / nr_channels;
        unsigned int chan = i % nr_channels;

        if (time < nr_timesteps) {

            // Load UVW
            float u = uvw[time_offset + time].u;
            float v = uvw[time_offset + time].v;
            float w = uvw[time_offset + time].w;

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

            // Iterate all pixels
            for (unsigned int y = 0; y < subgrid_size; y++) {
                for (unsigned int x = 0; x < subgrid_size; x++) {

                    // Compute l,m,n
                    const float l = compute_l(x, subgrid_size, image_size);
                    const float m = compute_m(y, subgrid_size, image_size);
                    const float n = compute_n(l, m);

                    // Compute phase offset
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                    // Iterate all terms
                    for (unsigned int term_nr = 0; term_nr < MAX_NR_TERMS; term_nr++) {

                        // Load pixels
                        unsigned int pixel_idx = index_pixels(subgrid_size, s, term_nr, 0, y, x);
                        float4 *pix_ptr = (float4 *) &scratch_pix[pixel_idx];
                        float2 pixelXX = make_float2(pix_ptr[0].x, pix_ptr[0].y);
                        float2 pixelXY = make_float2(pix_ptr[0].z, pix_ptr[0].w);
                        float2 pixelYX = make_float2(pix_ptr[1].x, pix_ptr[1].y);
                        float2 pixelYY = make_float2(pix_ptr[1].z, pix_ptr[1].w);

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
            for (unsigned int term_nr = 0; term_nr < (nr_terms+1); term_nr++) {
                const float scale = 1.0f / nr_pixels;
                unsigned int sum_idx = index_sums(nr_timesteps, nr_channels, (nr_terms+1), s, time, chan, 0, term_nr);
                float4 *sum_ptr = (float4 *) &scratch_sum[sum_idx];
                float4 sumA = make_float4(sumXX[term_nr].x, sumXX[term_nr].y, sumXY[term_nr].x, sumYX[term_nr].y);
                float4 sumB = make_float4(sumYX[term_nr].x, sumYX[term_nr].y, sumYY[term_nr].x, sumYY[term_nr].y);
                sum_ptr[0] = sumA * scale;
                sum_ptr[1] = sumB * scale;
            } // end for term_nr
        } // end if
    } // end for i (visibilities)

    __syncthreads();

    /*
        Phase 3: update local gradient and hessian
    */
    // Iterate all timesteps
    for (unsigned int time = 0; time < nr_timesteps; time++) {
        // Iterate all channels
        for (unsigned int chan = 0; chan < nr_channels; chan++) {

            __syncthreads();

            // Load sums for current visibility
            for (unsigned int term_nr = tid; term_nr < (nr_terms+1); term_nr += nr_threads) {
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    unsigned int idx = index_sums(nr_timesteps, nr_channels, (nr_terms+1), s, time, chan, pol, term_nr);
                    sums_[pol][term_nr] = scratch_sum[idx];
                }
            }

            __syncthreads();

            // Compute residual visibility
            float2 visibility_res[NR_POLARIZATIONS];
            for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                unsigned int time_idx = time_offset + time;
                unsigned int chan_idx = chan;
                unsigned int vis_idx  = index_visibility(nr_channels, time_idx, chan_idx, pol);
                visibility_res[pol] = visibilities[vis_idx + pol] - sums_[pol][nr_terms];
            }

            // Iterate all terms * terms
            for (unsigned int i = tid; i < (nr_terms*nr_terms); i += nr_threads) {
                unsigned term_nr0 = i / nr_terms;
                unsigned term_nr1 = i % nr_terms;

                // Iterate all polarizations
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {

                    // Update local gradient
                    if (i < nr_terms) {
                        gradient_[i].x +=
                            sums_[pol][i].x * visibility_res[pol].x +
                            sums_[pol][i].y * visibility_res[pol].y;
                        gradient_[i].y +=
                            sums_[pol][i].x * visibility_res[pol].y -
                            sums_[pol][i].y * visibility_res[pol].x;
                    }

                    // Update local hessian
                    if (i < (nr_terms*nr_terms)) {
                        hessian_[term_nr1][term_nr0].x +=
                            sums_[pol][term_nr0].x * sums_[pol][term_nr1].x +
                            sums_[pol][term_nr0].y * sums_[pol][term_nr1].y;
                        hessian_[term_nr0][term_nr1].y +=
                            sums_[pol][term_nr0].x * sums_[pol][term_nr1].y -
                            sums_[pol][term_nr0].y * sums_[pol][term_nr1].x;
                    }
                } // end for pol
            } // end for i (terms * terms)
        } // end for chan
    } // end for time

    __syncthreads();

    /*
        Phase 4: update global gradient and hessian
    */

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
} // end kernel_calibrate

} // end extern "C"
