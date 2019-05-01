#include <cmath>
#include <iostream>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

#pragma omp declare reduction(idgfloat2plus: idg::float2: omp_out += omp_in)

extern "C" {

void kernel_calibrate(
    const int                        nr_subgrids,
    const int                        grid_size,
    const int                        subgrid_size,
    const float                      image_size,
    const float                      w_step_in_lambda,
    const float* __restrict__        shift,
    const int                        nr_channels,
    const int                        nr_terms,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
          idg::float2*               visibilities,
    const idg::float2*               aterms,
    const idg::float2*               aterm_derivatives,
    const idg::Metadata*             metadata,
    const idg::float2*               subgrid,
    idg::float2*             hessian,
    idg::float2*             gradient)
{
    #if defined(USE_LOOKUP)
    CREATE_LOOKUP
    #endif

    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;

    // Compute l,m,n
    const unsigned nr_pixels = subgrid_size*subgrid_size;
    float l_[nr_pixels];
    float m_[nr_pixels];
    float n_[nr_pixels];

    for (unsigned i = 0; i < nr_pixels; i++) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;

        l_[i] = compute_l(x, subgrid_size, image_size);
        m_[i] = compute_m(y, subgrid_size, image_size);
        n_[i] = compute_n(l_[i], m_[i], shift);
    }

    // Initialize local gradient
    float gradient_real[nr_subgrids][nr_terms];
    float gradient_imag[nr_subgrids][nr_terms];
    size_t sizeof_gradient = nr_subgrids * nr_terms * sizeof(float);
    memset(gradient_real, 0, sizeof_gradient);
    memset(gradient_imag, 0, sizeof_gradient);

    // Initialize local hessian
    float hessian_real[nr_subgrids][nr_terms*nr_terms];
    float hessian_imag[nr_subgrids][nr_terms*nr_terms];
    size_t sizeof_hessian = nr_subgrids * nr_terms * nr_terms * sizeof(float);
    memset(hessian_real, 0, sizeof_hessian);
    memset(hessian_imag, 0, sizeof_hessian);

    // Iterate all subgrids
    #pragma omp parallel for schedule(guided)
    for (int s = 0; s < nr_subgrids; s++) {

        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int offset       = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Storage
        float pixels_xx_real[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_xy_real[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_yx_real[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_yy_real[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_xx_imag[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_xy_imag[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_yx_imag[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));
        float pixels_yy_imag[nr_pixels * (nr_terms+1)] __attribute__((aligned((ALIGNMENT))));

        // Apply aterm to subgrid
        for (unsigned i = 0; i < nr_pixels; i++) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Load aterm for station2
            size_t station2_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, 0, 0, station2, y, x);
            idg::float2 aXX2 = aterms[station2_idx + 0];
            idg::float2 aXY2 = aterms[station2_idx + 1];
            idg::float2 aYX2 = aterms[station2_idx + 2];
            idg::float2 aYY2 = aterms[station2_idx + 3];

            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            for (unsigned term_nr = 0; term_nr <= nr_terms; term_nr++) {
                // Load pixel values
                idg::float2 pixels[NR_POLARIZATIONS];
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    size_t src_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_src, x_src);
                    pixels[pol] = subgrid[src_idx];
                }

                idg::float2 aXX1;
                idg::float2 aXY1;
                idg::float2 aYX1;
                idg::float2 aYY1;

                if (term_nr == 0) {
                    // Load aterm for station1
                    size_t station1_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, 0, 0, station1, y, x);
                    aXX1 = aterms[station1_idx + 0];
                    aXY1 = aterms[station1_idx + 1];
                    aYX1 = aterms[station1_idx + 2];
                    aYY1 = aterms[station1_idx + 3];
                } else {
                    // Load aterm derivative
                    size_t station1_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, 0, 0, term_nr-1, y, x);
                    aXX1 = aterm_derivatives[station1_idx + 0];
                    aXY1 = aterm_derivatives[station1_idx + 1];
                    aYX1 = aterm_derivatives[station1_idx + 2];
                    aYY1 = aterm_derivatives[station1_idx + 3];
                }

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixels);

                // Store pixels
                pixels_xx_real[term_nr*nr_pixels + i] = pixels[0].real;
                pixels_xy_real[term_nr*nr_pixels + i] = pixels[1].real;
                pixels_yx_real[term_nr*nr_pixels + i] = pixels[2].real;
                pixels_yy_real[term_nr*nr_pixels + i] = pixels[3].real;
                pixels_xx_imag[term_nr*nr_pixels + i] = pixels[0].imag;
                pixels_xy_imag[term_nr*nr_pixels + i] = pixels[1].imag;
                pixels_yx_imag[term_nr*nr_pixels + i] = pixels[2].imag;
                pixels_yy_imag[term_nr*nr_pixels + i] = pixels[3].imag;
            } // end for terms
        } // end for pixels

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2)
                               * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2)
                               * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        float phase_offset[nr_pixels];

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = uvw[offset + time].u;
            float v = uvw[offset + time].v;
            float w = uvw[offset + time].w;

            float phase_index[nr_pixels];

            for (unsigned i = 0; i < nr_pixels; i++) {
                // Compute phase index
                phase_index[i] = u*l_[i] + v*m_[i] + w*n_[i];

                // Compute phase offset
                if (time == 0) {
                    phase_offset[i] = u_offset*l_[i] + v_offset*m_[i] + w_offset*n_[i];
                }
            }

            // Iterate all channels
            for (int chan = 0; chan < nr_channels; chan++) {
                // Compute phase
                float phase[nr_pixels];

                for (unsigned i = 0; i < nr_pixels; i++) {
                    // Compute phase
                    float wavenumber = wavenumbers[chan];
                    phase[i] = (phase_index[i] * wavenumber) - phase_offset[i];
                }

                // Compute phasor
                float phasor_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));;
                float phasor_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));;
                #if defined(USE_LOOKUP)
                compute_sincos(nr_pixels, phase, lookup, phasor_imag, phasor_real);
                #else
                compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);
                #endif

                // Compute visibilities
                idg::float2 sums[NR_POLARIZATIONS * (nr_terms + 1)] = {};

                for (int term_nr = 0; term_nr <= nr_terms; term_nr++) {
                    compute_reduction(
                        nr_pixels,
                        pixels_xx_real + term_nr*nr_pixels, pixels_xy_real + term_nr*nr_pixels,
                        pixels_yx_real + term_nr*nr_pixels, pixels_yy_real + term_nr*nr_pixels,
                        pixels_xx_imag + term_nr*nr_pixels, pixels_xy_imag + term_nr*nr_pixels,
                        pixels_yx_imag + term_nr*nr_pixels, pixels_yy_imag + term_nr*nr_pixels,
                        phasor_real, phasor_imag, sums + term_nr*NR_POLARIZATIONS);
                }

                // Scale visibilities
                const float scale = 1.0f / nr_pixels;
                for (int i = 0; i < (nr_terms + 1) * NR_POLARIZATIONS; i++) {
                    sums[i].real *= scale;
                    sums[i].imag *= scale;
                }

                // Store visibilities
                int time_idx = offset + time;
                int chan_idx = chan;
                size_t vis_idx = index_visibility( nr_channels, NR_POLARIZATIONS, time_idx, chan_idx, 0);

                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    // Compute residual visibilities
                    sums[pol].real = visibilities[vis_idx+pol].real - sums[pol].real;
                    sums[pol].imag = visibilities[vis_idx+pol].imag - sums[pol].imag;

                    // Update local gradient
                    for (int term_nr0 = 1; term_nr0 <= nr_terms; term_nr0++) {
                        gradient_real[s][term_nr0-1] +=
                           sums[pol + term_nr0*NR_POLARIZATIONS].real * sums[pol].real +
                           sums[pol + term_nr0*NR_POLARIZATIONS].imag * sums[pol].imag;
                        gradient_imag[s][term_nr0-1] +=
                           sums[pol + term_nr0*NR_POLARIZATIONS].real * sums[pol].imag -
                           sums[pol + term_nr0*NR_POLARIZATIONS].imag * sums[pol].real;

                        // Update local hessian
                        for (int term_nr1 = 1; term_nr1 <= nr_terms; term_nr1++) {
                            hessian_real[s][(term_nr1-1)*nr_terms + term_nr0-1] +=
                                sums[pol + term_nr0*NR_POLARIZATIONS].real * sums[pol + term_nr1*NR_POLARIZATIONS].real +
                                sums[pol + term_nr0*NR_POLARIZATIONS].imag * sums[pol + term_nr1*NR_POLARIZATIONS].imag;
                            hessian_imag[s][(term_nr1-1)*nr_terms + term_nr0-1] +=
                                sums[pol + term_nr0*NR_POLARIZATIONS].real * sums[pol + term_nr1*NR_POLARIZATIONS].imag -
                                sums[pol + term_nr0*NR_POLARIZATIONS].imag * sums[pol + term_nr1*NR_POLARIZATIONS].real;
                        } // end for term1
                    } // end for term0
                } // end for polarization
            } // end for channel
        } // end for time
    } // end #pragma parallel

    // Update global gradient
    for (int s = 0; s < nr_subgrids; s++) {
        for (unsigned i = 0; i < nr_terms; i++) {
            gradient[i].real += gradient_real[s][i];
            gradient[i].imag += gradient_imag[s][i];
        }
    }

    // Update global hessian
    for (int s = 0; s < nr_subgrids; s++) {
        for (unsigned i = 0; i < nr_terms*nr_terms; i++) {
            hessian[i].real += hessian_real[s][i];
            hessian[i].imag += hessian_imag[s][i];
        }
    }

} // end kernel_calibrate

} // end extern "C"
