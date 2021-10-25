// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/memory.h"
#include "common/Types.h"
#include "common/Index.h"

#include "Math.h"

#if not defined(__PPC__)
inline size_t index_phasors(unsigned int nr_timesteps, unsigned int nr_channels,
                            unsigned int subgrid_size, unsigned int subgrid,
                            unsigned int time, unsigned int chan,
                            unsigned int pixel) {
  // phasor: [nr_subgrids][nr_time][nr_channels][nr_pixels]
  return static_cast<size_t>(subgrid) * nr_timesteps * nr_channels *
             subgrid_size * subgrid_size +
         static_cast<size_t>(time) * nr_channels * subgrid_size * subgrid_size +
         static_cast<size_t>(chan) * subgrid_size * subgrid_size +
         static_cast<size_t>(pixel);
}

namespace idg {
namespace kernel {
namespace cpu {
namespace optimized {

void kernel_calibrate(
    const unsigned int nr_subgrids, const unsigned int nr_polarizations,
    const unsigned long grid_size, const unsigned int subgrid_size,
    const float image_size, const float w_step_in_lambda,
    const float* __restrict__ shift, const unsigned int max_nr_timesteps,
    const unsigned int nr_channels, const unsigned int nr_stations,
    const unsigned int nr_terms, const unsigned int nr_time_slots,
    const idg::UVW<float>* uvw, const float* wavenumbers,
    std::complex<float>* visibilities, const float* weights,
    const std::complex<float>* aterms,
    const std::complex<float>* aterm_derivatives, const int* aterms_indices,
    const idg::Metadata* metadata, const std::complex<float>* subgrid,
    const std::complex<float>* phasors, double* hessian, double* gradient,
    double* residual) {
#if defined(USE_LOOKUP)
  initialize_lookup();
#endif

  // Initialize local residual
  double residual_local[nr_subgrids] __attribute__((aligned((ALIGNMENT))));
  size_t sizeof_residual = nr_subgrids * sizeof(double);
  memset(residual_local, 0, sizeof_residual);

  // Initialize local gradient
  double gradient_local[nr_subgrids][nr_time_slots][nr_terms]
      __attribute__((aligned((ALIGNMENT))));
  size_t sizeof_gradient =
      nr_subgrids * nr_time_slots * nr_terms * sizeof(double);
  memset(gradient_local, 0, sizeof_gradient);

  // Initialize local hessian
  double hessian_local[nr_subgrids][nr_time_slots][nr_terms][nr_terms]
      __attribute__((aligned((ALIGNMENT))));
  size_t sizeof_hessian =
      nr_subgrids * nr_time_slots * nr_terms * nr_terms * sizeof(double);
  memset(hessian_local, 0, sizeof_hessian);

// Iterate all subgrids
#pragma omp parallel for schedule(guided)
  for (unsigned int s = 0; s < nr_subgrids; s++) {
    // Load metadata
    const idg::Metadata m = metadata[s];
    const unsigned int time_offset = m.time_index;
    const unsigned int nr_timesteps = m.nr_timesteps;
    const unsigned int channel_begin = m.channel_begin;
    const unsigned int channel_end = m.channel_end;
    const unsigned int station1 = m.baseline.station1;
    const unsigned int station2 = m.baseline.station2;

    // Initialize aterm index to first timestep
    size_t aterm_idx_previous = aterms_indices[time_offset];

    // Storage
    unsigned nr_pixels = subgrid_size * subgrid_size;
    float pixels_xx_real[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_xy_real[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_yx_real[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_yy_real[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_xx_imag[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_xy_imag[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_yx_imag[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));
    float pixels_yy_imag[(nr_terms + 1)][nr_pixels]
        __attribute__((aligned((ALIGNMENT))));

    // Iterate all timesteps
    for (unsigned int time = 0; time < nr_timesteps; time++) {
      // Get aterm indices for current timestep
      size_t aterm_idx_current = aterms_indices[time_offset + time];

      // Determine whether aterm has changed
      bool aterm_changed = aterm_idx_previous != aterm_idx_current;

      if (time == 0 || aterm_changed) {
        // Apply aterm to subgrid
        for (unsigned term_nr = 0; term_nr < (nr_terms + 1); term_nr++) {
          for (unsigned i = 0; i < nr_pixels; i++) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size / 2)) % subgrid_size;
            int y_src = (y + (subgrid_size / 2)) % subgrid_size;

            // Load pixel values
            std::complex<float> pixels[nr_polarizations];
            for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
              size_t src_idx = index_subgrid(nr_polarizations, subgrid_size, s,
                                             pol, y_src, x_src);
              pixels[pol] = subgrid[src_idx];
            }

            // Get pointer to first aterm
            std::complex<float>* aterm1_ptr;

            if (term_nr == nr_terms) {
              unsigned int station1_idx =
                  index_aterm(subgrid_size, nr_polarizations, nr_stations,
                              aterm_idx_current, station1, y, x, 0);
              aterm1_ptr = (std::complex<float>*)&aterms[station1_idx];
            } else {
              unsigned int station1_idx =
                  index_aterm(subgrid_size, nr_polarizations, nr_terms,
                              aterm_idx_current, term_nr, y, x, 0);
              aterm1_ptr =
                  (std::complex<float>*)&aterm_derivatives[station1_idx];
            }

            // Get pointer to second aterm
            unsigned int station2_idx =
                index_aterm(subgrid_size, nr_polarizations, nr_stations,
                            aterm_idx_current, station2, y, x, 0);
            std::complex<float>* aterm2_ptr =
                (std::complex<float>*)&aterms[station2_idx];

            // Apply aterm
            apply_aterm_degridder(pixels, aterm1_ptr, aterm2_ptr);

            // Store pixels
            pixels_xx_real[term_nr][i] = pixels[0].real();
            pixels_xy_real[term_nr][i] = pixels[1].real();
            pixels_yx_real[term_nr][i] = pixels[2].real();
            pixels_yy_real[term_nr][i] = pixels[3].real();
            pixels_xx_imag[term_nr][i] = pixels[0].imag();
            pixels_xy_imag[term_nr][i] = pixels[1].imag();
            pixels_yx_imag[term_nr][i] = pixels[2].imag();
            pixels_yy_imag[term_nr][i] = pixels[3].imag();
          }  // end for terms
        }    // end for pixels

        // Update aterm index
        aterm_idx_previous = aterm_idx_current;
      }

      // Iterate all channels
      for (unsigned int chan = channel_begin; chan < channel_end; chan++) {
        // Load phasor
        float phasor_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float phasor_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        for (unsigned i = 0; i < nr_pixels; i++) {
          unsigned idx = index_phasors(max_nr_timesteps, nr_channels,
                                       subgrid_size, s, time, chan, i);
          std::complex<float> phasor = phasors[idx];
          phasor_real[i] = phasor.real();
          phasor_imag[i] = phasor.imag();
        }

        // Compute visibilities
        float sums_real[nr_polarizations][nr_terms + 1];
        float sums_imag[nr_polarizations][nr_terms + 1];

        for (unsigned int term_nr = 0; term_nr < (nr_terms + 1); term_nr++) {
          std::complex<float> sum[nr_polarizations]
              __attribute__((aligned(ALIGNMENT)));

          compute_reduction(nr_pixels, pixels_xx_real[term_nr],
                            pixels_xy_real[term_nr], pixels_yx_real[term_nr],
                            pixels_yy_real[term_nr], pixels_xx_imag[term_nr],
                            pixels_xy_imag[term_nr], pixels_yx_imag[term_nr],
                            pixels_yy_imag[term_nr], phasor_real, phasor_imag,
                            sum);

          // Store and scale sums
          const float scale = 1.0f / nr_pixels;
          for (unsigned pol = 0; pol < nr_polarizations; pol++) {
            sums_real[pol][term_nr] = sum[pol].real() * scale;
            sums_imag[pol][term_nr] = sum[pol].imag() * scale;
          }
        }

        // Compute residual visibilities
        float visibility_res_real[nr_polarizations];
        float visibility_res_imag[nr_polarizations];
        for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
          int time_idx = time_offset + time;
          int chan_idx = chan;
          size_t vis_idx = index_visibility(nr_polarizations, nr_channels,
                                            time_idx, chan_idx, pol);
          visibility_res_real[pol] =
              visibilities[vis_idx].real() - sums_real[pol][nr_terms];
          visibility_res_imag[pol] =
              visibilities[vis_idx].imag() - sums_imag[pol][nr_terms];
        }

        // Update local residual and gradient
        for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
          int time_idx = time_offset + time;
          int chan_idx = chan;
          size_t vis_idx = index_visibility(nr_polarizations, nr_channels,
                                            time_idx, chan_idx, pol);
          residual_local[s] +=
              weights[vis_idx] *
              (visibility_res_real[pol] * visibility_res_real[pol] +
               visibility_res_imag[pol] * visibility_res_imag[pol]);
          for (unsigned int term_nr0 = 0; term_nr0 < nr_terms; term_nr0++) {
            gradient_local[s][aterm_idx_current][term_nr0] +=
                weights[vis_idx] *
                (sums_real[pol][term_nr0] * visibility_res_real[pol] +
                 sums_imag[pol][term_nr0] * visibility_res_imag[pol]);
          }
        }

        // Update local hessian
        for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
          int time_idx = time_offset + time;
          int chan_idx = chan;
          size_t vis_idx = index_visibility(nr_polarizations, nr_channels,
                                            time_idx, chan_idx, pol);
          for (unsigned int term_nr1 = 0; term_nr1 < nr_terms; term_nr1++) {
            for (unsigned int term_nr0 = 0; term_nr0 < nr_terms; term_nr0++) {
              hessian_local[s][aterm_idx_current][term_nr1][term_nr0] +=
                  weights[vis_idx] *
                  (sums_real[pol][term_nr0] * sums_real[pol][term_nr1] +
                   sums_imag[pol][term_nr0] * sums_imag[pol][term_nr1]);
            }
          }
        }
      }  // end for channel
    }    // end for time
  }      // end #pragma parallel

  // Update global residual
  for (unsigned int s = 0; s < nr_subgrids; s++) {
    *residual += residual_local[s];
  }

  // Update global gradient
  for (unsigned int s = 0; s < nr_subgrids; s++) {
    for (unsigned int aterm_idx = 0; aterm_idx < nr_time_slots; aterm_idx++) {
      for (unsigned int i = 0; i < nr_terms; i++) {
        unsigned idx = aterm_idx * nr_terms + i;
        gradient[idx] += gradient_local[s][aterm_idx][i];
      }
    }
  }

  // Update global hessian
  for (unsigned int s = 0; s < nr_subgrids; s++) {
    for (unsigned int aterm_idx = 0; aterm_idx < nr_time_slots; aterm_idx++) {
      for (unsigned int term_nr1 = 0; term_nr1 < nr_terms; term_nr1++) {
        for (unsigned int term_nr0 = 0; term_nr0 < nr_terms; term_nr0++) {
          unsigned idx =
              aterm_idx * nr_terms * nr_terms + term_nr1 * nr_terms + term_nr0;
          hessian[idx] += hessian_local[s][aterm_idx][term_nr1][term_nr0];
        }
      }
    }
  }

}  // end kernel_calibrate

void kernel_phasor(const int nr_subgrids, const long grid_size,
                   const int subgrid_size, const float image_size,
                   const float w_step_in_lambda,
                   const float* __restrict__ shift, const int max_nr_timesteps,
                   const int nr_channels, const idg::UVW<float>* uvw,
                   const float* wavenumbers, const idg::Metadata* metadata,
                   std::complex<float>* phasors) {
#if defined(USE_LOOKUP)
  initialize_lookup();
#endif

  // Compute l,m,n
  const unsigned nr_pixels = subgrid_size * subgrid_size;
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
    const idg::Metadata m = metadata[s];
    const int time_offset = m.time_index;
    const int nr_timesteps = m.nr_timesteps;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
    const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

    // Compute u and v offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float w_offset = 2 * M_PI * w_offset_in_lambda;

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
        phase_index[i] = u * l_[i] + v * m_[i] + w * n_[i];

        // Compute phase offset
        if (time == 0) {
          phase_offset[i] =
              u_offset * l_[i] + v_offset * m_[i] + w_offset * n_[i];
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
        float phasor_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        ;
        float phasor_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        ;
        compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);

        // Store phasor
        for (unsigned i = 0; i < nr_pixels; i++) {
          unsigned idx = index_phasors(max_nr_timesteps, nr_channels,
                                       subgrid_size, s, time, chan, i);
          phasors[idx] = {phasor_real[i], phasor_imag[i]};
        }
      }  // end for channel
    }    // end for time
  }      // end #pragma parallel
}  // end kernel_phasor

#endif

}  // end namespace optimized
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg