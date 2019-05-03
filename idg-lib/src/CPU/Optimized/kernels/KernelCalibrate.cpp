#include <cmath>
#include <iostream>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

inline size_t index_phasors(
    unsigned int nr_timesteps,
    unsigned int nr_channels,
    unsigned int subgrid_size,
    unsigned int subgrid,
    unsigned int time,
    unsigned int chan,
    unsigned int pixel)
{
    // phasor: [nr_subgrids][nr_time][nr_channels][nr_pixels]
    return static_cast<size_t>(subgrid) * nr_timesteps * nr_channels * subgrid_size * subgrid_size +
           static_cast<size_t>(time) * nr_channels * subgrid_size * subgrid_size +
           static_cast<size_t>(chan) * subgrid_size * subgrid_size +
           static_cast<size_t>(pixel);
}

extern "C" {

void kernel_calibrate(
    const unsigned int               nr_subgrids,
    const unsigned int               grid_size,
    const unsigned int               subgrid_size,
    const float                      image_size,
    const float                      w_step_in_lambda,
    const float* __restrict__        shift,
    const unsigned int               nr_channels,
    const unsigned int               nr_terms,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
          idg::float2*               visibilities,
    const idg::float2*               aterms,
    const idg::float2*               aterm_derivatives,
    const idg::Metadata*             metadata,
    const idg::float2*               subgrid,
    const idg::float2*               phasors,
    idg::float2*                     hessian,
    idg::float2*                     gradient)
{
    #if defined(USE_LOOKUP)
    CREATE_LOOKUP
    #endif

    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;

    #define NR_TERMS 8
    assert(nr_terms <= NR_TERMS);

    // Initialize local gradient
    float gradient_real[nr_subgrids][NR_TERMS];
    float gradient_imag[nr_subgrids][NR_TERMS];
    size_t sizeof_gradient = nr_subgrids * NR_TERMS * sizeof(float);
    memset(gradient_real, 0, sizeof_gradient);
    memset(gradient_imag, 0, sizeof_gradient);

    // Initialize local hessian
    float hessian_real[nr_subgrids][NR_TERMS][NR_TERMS];
    float hessian_imag[nr_subgrids][NR_TERMS][NR_TERMS];
    size_t sizeof_hessian = nr_subgrids * NR_TERMS * NR_TERMS * sizeof(float);
    memset(hessian_real, 0, sizeof_hessian);
    memset(hessian_imag, 0, sizeof_hessian);

    // Iterate all subgrids
    #pragma omp parallel for schedule(guided)
    for (unsigned int s = 0; s < nr_subgrids; s++) {

        // Load metadata
        const idg::Metadata m  = metadata[s];
        const unsigned int time_offset  = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const unsigned int nr_timesteps = m.nr_timesteps;
        const unsigned int station1     = m.baseline.station1;
        const unsigned int station2     = m.baseline.station2;

        // Storage
        unsigned nr_pixels = subgrid_size*subgrid_size;
        float pixels_xx_real[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_xy_real[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yx_real[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yy_real[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_xx_imag[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_xy_imag[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yx_imag[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yy_imag[(nr_terms+1)][nr_pixels] __attribute__((aligned((ALIGNMENT))));

        // Apply aterm to subgrid
        for (unsigned term_nr = 0; term_nr <= nr_terms; term_nr++) {
            for (unsigned i = 0; i < nr_pixels; i++) {
                int y = i / subgrid_size;
                int x = i % subgrid_size;

                // Compute shifted position in subgrid
                int x_src = (x + (subgrid_size/2)) % subgrid_size;
                int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Load pixel values
                idg::float2 pixels[NR_POLARIZATIONS];
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    size_t src_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_src, x_src);
                    pixels[pol] = subgrid[src_idx];
                }

                // Load first aterm
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

                // Load aterm for station2
                size_t station2_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, 0, 0, station2, y, x);
                idg::float2 aXX2 = aterms[station2_idx + 0];
                idg::float2 aXY2 = aterms[station2_idx + 1];
                idg::float2 aYX2 = aterms[station2_idx + 2];
                idg::float2 aYY2 = aterms[station2_idx + 3];

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixels);

                // Store pixels
                pixels_xx_real[term_nr][i] = pixels[0].real;
                pixels_xy_real[term_nr][i] = pixels[1].real;
                pixels_yx_real[term_nr][i] = pixels[2].real;
                pixels_yy_real[term_nr][i] = pixels[3].real;
                pixels_xx_imag[term_nr][i] = pixels[0].imag;
                pixels_xy_imag[term_nr][i] = pixels[1].imag;
                pixels_yx_imag[term_nr][i] = pixels[2].imag;
                pixels_yy_imag[term_nr][i] = pixels[3].imag;
            } // end for terms
        } // end for pixels

        // Iterate all timesteps
        for (unsigned int time = 0; time < nr_timesteps; time++) {
            // Iterate all channels
            for (unsigned int chan = 0; chan < nr_channels; chan++) {
                // Load phasor
                float phasor_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
                float phasor_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
                for (unsigned i = 0; i < nr_pixels; i++) {
                    unsigned idx = index_phasors(nr_timesteps, nr_channels, subgrid_size, s, time, chan, i);
                    idg::float2 phasor = phasors[idx];
                    phasor_real[i] = phasor.real;
                    phasor_imag[i] = phasor.imag;
                }

                // Compute visibilities
                float sums_real[NR_POLARIZATIONS][nr_terms+1];
                float sums_imag[NR_POLARIZATIONS][nr_terms+1];

                for (unsigned int term_nr = 0; term_nr <= nr_terms; term_nr++) {
                    idg::float2 sum[NR_POLARIZATIONS];

                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        sum[pol].real = 0;
                        sum[pol].imag = 0;
                    }

                    compute_reduction(
                        nr_pixels,
                        pixels_xx_real[term_nr], pixels_xy_real[term_nr],
                        pixels_yx_real[term_nr], pixels_yy_real[term_nr],
                        pixels_xx_imag[term_nr], pixels_xy_imag[term_nr],
                        pixels_yx_imag[term_nr], pixels_yy_imag[term_nr],
                        phasor_real, phasor_imag, sum);

                    // Store and scale sums
                    const float scale = term_nr < nr_terms ? 1.0f / nr_pixels : 1.0f;
                    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        sums_real[pol][term_nr] = sum[pol].real * scale;
                        sums_imag[pol][term_nr] = sum[pol].imag * scale;
                    }
                }

                // Store visibilities
                int time_idx = time_offset + time;
                int chan_idx = chan;
                size_t vis_idx = index_visibility( nr_channels, NR_POLARIZATIONS, time_idx, chan_idx, 0);

                // Compute residual visibilities
                float visibility_res_real[NR_POLARIZATIONS];
                float visibility_res_imag[NR_POLARIZATIONS];
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    visibility_res_real[pol] = visibilities[vis_idx+pol].real - sums_real[pol][0];
                    visibility_res_imag[pol] = visibilities[vis_idx+pol].imag - sums_imag[pol][0];
                }

                // Update local gradient
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    for (unsigned int term_nr0 = 0; term_nr0 < nr_terms; term_nr0++) {
                        gradient_real[s][term_nr0] +=
                           sums_real[pol][term_nr0+1] * visibility_res_real[pol] +
                           sums_imag[pol][term_nr0+1] * visibility_res_imag[pol];
                        gradient_imag[s][term_nr0] +=
                           sums_real[pol][term_nr0+1] * visibility_res_imag[pol] -
                           sums_imag[pol][term_nr0+1] * visibility_res_real[pol];
                    }
                }

                // Update local hessian
                for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    for (unsigned int term_nr1 = 0; term_nr1 < nr_terms; term_nr1++) {
                        for (unsigned int term_nr0 = 0; term_nr0 < nr_terms; term_nr0++) {
                            hessian_real[s][term_nr1][term_nr0] +=
                                sums_real[pol][term_nr0+1] * sums_real[pol][term_nr1+1] +
                                sums_imag[pol][term_nr0+1] * sums_imag[pol][term_nr1+1];
                            hessian_imag[s][term_nr1][term_nr0] +=
                                sums_real[pol][term_nr0+1] * sums_imag[pol][term_nr1+1] -
                                sums_imag[pol][term_nr0+1] * sums_real[pol][term_nr1+1];
                        }
                    }
                }
            } // end for channel
        } // end for time
    } // end #pragma parallel

    // Update global gradient
    for (unsigned int s = 0; s < nr_subgrids; s++) {
        for (unsigned int i = 0; i < nr_terms; i++) {
            gradient[i].real += gradient_real[s][i];
            gradient[i].imag += gradient_imag[s][i];
        }
    }

    // Update global hessian
    for (unsigned int s = 0; s < nr_subgrids; s++) {
        for (unsigned int term_nr1 = 0; term_nr1 < nr_terms; term_nr1++) {
            for (unsigned int term_nr0 = 0; term_nr0 < nr_terms; term_nr0++) {
                unsigned idx = term_nr1 * nr_terms + term_nr0;
                hessian[idx].real += hessian_real[s][term_nr1][term_nr0];
                hessian[idx].imag += hessian_imag[s][term_nr1][term_nr0];
            }
        }
    }

} // end kernel_calibrate

void kernel_phasor(
    const int                        nr_subgrids,
    const int                        grid_size,
    const int                        subgrid_size,
    const float                      image_size,
    const float                      w_step_in_lambda,
    const float* __restrict__        shift,
    const int                        nr_channels,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
    const idg::Metadata*             metadata,
          idg::float2*               phasors)
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

    // Iterate all subgrids
    #pragma omp parallel for schedule(guided)
    for (int s = 0; s < nr_subgrids; s++) {

        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int time_offset  = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

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
            float u = uvw[time_offset + time].u;
            float v = uvw[time_offset + time].v;
            float w = uvw[time_offset + time].w;

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
                    phase[i] = (phase_index[i] * wavenumbers[chan]) - phase_offset[i];
                }

                // Compute phasor
                float phasor_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));;
                float phasor_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));;
                #if defined(USE_LOOKUP)
                compute_sincos(nr_pixels, phase, lookup, phasor_imag, phasor_real);
                #else
                compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);
                #endif

                // Store phasor
                for (unsigned i = 0; i < nr_pixels; i++) {
                    unsigned idx = index_phasors(nr_timesteps, nr_channels, subgrid_size, s, time, chan, i);
                    phasors[idx] = { phasor_real[i], phasor_imag[i] };
                }
            } // end for channel
        } // end for time
    } // end #pragma parallel
} // end kernel_phasor

} // end extern "C"
