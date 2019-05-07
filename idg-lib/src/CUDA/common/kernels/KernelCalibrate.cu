#include "Types.h"
#include "math.cu"

#define MAX_NR_TERMS 8

extern "C" {

__global__ void kernel_calibrate(
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
    const unsigned nr_stations  = 0;
    const unsigned aterm_index  = 0;

    /*
        Phase 1: apply aterm to subgrids and store prepared subgrid in device memory
    */

    // Apply aterm to subgrid
    for (unsigned i = tid; i < nr_pixels; i += nr_threads) {
        for (unsigned term_nr = 0; term_nr <= nr_terms; term_nr++) {
            unsigned y = i / subgrid_size;
            unsigned x = i % subgrid_size;

            // Compute shifted position in subgrid
            unsigned x_src = (x + (subgrid_size/2)) % subgrid_size;
            unsigned y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Load pixels
            unsigned subgrid_idx = s;
            unsigned pixel_idx_xx = index_subgrid(subgrid_size, subgrid_idx, 0, y_src, x_src);
            unsigned pixel_idx_xy = index_subgrid(subgrid_size, subgrid_idx, 1, y_src, x_src);
            unsigned pixel_idx_yx = index_subgrid(subgrid_size, subgrid_idx, 2, y_src, x_src);
            unsigned pixel_idx_yy = index_subgrid(subgrid_size, subgrid_idx, 3, y_src, x_src);
            float2 pixelXX = subgrid[pixel_idx_xx];
            float2 pixelXY = subgrid[pixel_idx_xy];
            float2 pixelYX = subgrid[pixel_idx_yx];
            float2 pixelYY = subgrid[pixel_idx_yy];

            // Load first aterm
            float2 aXX1, aXY1, aYX1, aYY1;

            if (term_nr == nr_terms) {
                // Load aterm for station1
                read_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, aterm, &aXX1, &aXY1, &aYX1, &aYY1);
            } else {
                // Load aterm derivative
                read_aterm(subgrid_size, nr_stations, aterm_index, term_nr, y, x, aterm, &aXX1, &aXY1, &aYX1, &aYY1);
            }

            // Load second aterm
            float2 aXX2, aXY2, aYX2, aYY2;
            read_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, aterm, &aXX2, &aXY2, &aYX2, &aYY2);

            // Apply aterm
            apply_aterm(
                aXX1, aXY1, aYX1, aYY1,
                aXX2, aXY2, aYX2, aYY2,
                pixelXX, pixelXY, pixelYX, pixelYY);

            // Store pixels
            subgrid_idx = s * nr_terms + term_nr;
            pixel_idx_xx = index_subgrid(subgrid_size, subgrid_idx, 0, y_src, x_src);
            pixel_idx_xy = index_subgrid(subgrid_size, subgrid_idx, 1, y_src, x_src);
            pixel_idx_yx = index_subgrid(subgrid_size, subgrid_idx, 2, y_src, x_src);
            pixel_idx_yy = index_subgrid(subgrid_size, subgrid_idx, 3, y_src, x_src);
            scratch_pix[pixel_idx_xx] = pixelXX;
            scratch_pix[pixel_idx_xy] = pixelXY;
            scratch_pix[pixel_idx_yx] = pixelYX;
            scratch_pix[pixel_idx_yy] = pixelYY;
        } // end for terms
    } // end for pixels

    __syncthreads();

    __shared__ float2 sums_[NR_POLARIZATIONS][MAX_NR_TERMS];
    __shared__ float2 gradient_[MAX_NR_TERMS];
    __shared__ float2 hessian_[MAX_NR_TERMS][MAX_NR_TERMS];

    // Initialize local gradient and hessian to zero
    for (unsigned int i = tid; i < nr_terms * nr_terms; i += nr_threads) {
        unsigned term_nr1 = i / nr_terms;
        unsigned term_nr0 = i % nr_terms;

        if (term_nr1 == 0) {
            gradient_[term_nr0] = make_float2(0, 0);
        }

        hessian_[term_nr1][term_nr0] = make_float2(0, 0);
    } // end for i

    // Iterate all timesteps
    for (unsigned int time = 0; time < nr_timesteps; time++) {

        // Iterate all channels
        for (unsigned int chan = 0; chan < nr_channels; chan++) {

            /*
                Phase 2: "degrid" all prepared subgrids, store results in local memory
            */

            // Iterate all terms and polarizations
            for (unsigned int i = tid; i < nr_terms * NR_POLARIZATIONS; i += nr_threads) {
                unsigned term_nr = i / nr_terms;
                unsigned pol     = i % nr_terms;
                float2 sum = make_float2(0, 0);

                // Iterate all pixels
                for (unsigned int j = 0; j < nr_pixels; j++) {
                    unsigned y = j / subgrid_size;
                    unsigned x = j % subgrid_size;

                    // Compute phasor
                    float2 phasor = make_float2(1, 0); // TODO

                    // Load pixel
                    unsigned subgrid_idx = s * nr_terms + term_nr;
                    unsigned pixel_idx   = index_subgrid(subgrid_size, subgrid_idx, pol, y, x);
                    float2 pixel = scratch_pix[pixel_idx];

                    sum.x += phasor.x * pixel.x;
                    sum.y += phasor.x * pixel.y;
                    sum.x -= phasor.y * pixel.y;
                    sum.y += phasor.y * pixel.x;
                } // end for j (pixels)

                // Scale sums
                const float scale = 1.0f / nr_pixels;
                sums_[pol][term_nr] = sum * scale;

            } // end for i (terms and polarizations)

            __syncthreads();

            /*
                Phase 3: update local gradient and hessian
            */

            // Compute residual visibility
            float2 visibility_res[NR_POLARIZATIONS];
            for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int time_idx = time_offset + time;
                int chan_idx = chan;
                unsigned vis_idx = index_visibility(nr_channels, time_idx, chan_idx, pol);
                visibility_res[pol] = visibilities[vis_idx + pol] - sums_[pol][nr_terms];
            }

            // Iterate all terms * terms
            for (unsigned int i = 0; i < nr_terms * nr_terms; i += nr_threads) {
                unsigned term_nr1 = i / nr_terms;
                unsigned term_nr0 = i % nr_terms;

                // Iterate all polarizations
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {

                    // Update local gradient
                    if (i < nr_terms) {
                        gradient_[term_nr0].x +=
                           sums_[pol][term_nr0].x * visibility_res[pol].x +
                           sums_[pol][term_nr0].y * visibility_res[pol].y;
                        gradient_[term_nr0].y +=
                           sums_[pol][term_nr0].x * visibility_res[pol].y -
                           sums_[pol][term_nr0].y * visibility_res[pol].x;
                    }

                    // Update local hessian
                    if (i < nr_terms * nr_terms) {
                        hessian_[term_nr1][term_nr0].x +=
                            sums_[pol][term_nr0].x * sums_[pol][term_nr1].x +
                            sums_[pol][term_nr0].y * sums_[pol][term_nr1].y;
                        hessian_[term_nr1][term_nr0].y +=
                            sums_[pol][term_nr0].x * sums_[pol][term_nr1].y -
                            sums_[pol][term_nr0].y * sums_[pol][term_nr1].x;
                    }
                } // end for pol
            } // end for i (terms * terms)
        } // end for chan
    } // end for time

    __syncthreads();

    /*
        Phase 4:  update global gradient and hessian
    */

    // Iterate all terms * terms
    for (unsigned int i = tid; i < nr_terms * nr_terms; i += nr_threads) {
        unsigned term_nr1 = i / nr_terms;
        unsigned term_nr0 = i % nr_terms;

        if (i < nr_terms) {
            atomicAdd(&gradient[i], gradient_[term_nr0]);
        }

        if (i < nr_terms * nr_terms) {
            atomicAdd(&hessian[i], hessian_[term_nr1][term_nr0]);
        }
    } // end for i
} // end kernel_calibrate

} // end extern "C"
