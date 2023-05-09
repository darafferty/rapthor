// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/memory.h"
#include "common/Types.h"
#include "common/Index.h"

#include "Math.h"

namespace idg {
namespace kernel {
namespace cpu {
namespace optimized {

void kernel_degridder(
    const int nr_subgrids, const int nr_polarizations, const long grid_size,
    const int subgrid_size, const float image_size,
    const float w_step_in_lambda, const float* __restrict__ shift,
    const int nr_correlations, const int nr_channels, const int nr_stations,
    const idg::UVW<float>* uvw, const float* wavenumbers,
    std::complex<float>* visibilities, const float* taper,
    const std::complex<float>* aterms, const unsigned int* aterm_indices,
    const idg::Metadata* metadata, const std::complex<float>* subgrid) {
#if defined(USE_LOOKUP)
  initialize_lookup();
#endif

  // Compute l,m,n
  const unsigned nr_pixels = subgrid_size * subgrid_size;
  float l_offset[nr_pixels];
  float m_offset[nr_pixels];
  float n_offset[nr_pixels];
  float l_index[nr_pixels];
  float m_index[nr_pixels];
  float n_index[nr_pixels];

  for (unsigned i = 0; i < nr_pixels; i++) {
    int y = i / subgrid_size;
    int x = i % subgrid_size;

    l_offset[i] = compute_l(x, subgrid_size, image_size);
    m_offset[i] = compute_m(y, subgrid_size, image_size);
    l_index[i] = l_offset[i] + shift[0];
    m_index[i] = m_offset[i] - shift[1];
    n_index[i] = compute_n(l_index[i], m_index[i]);
    n_offset[i] = n_index[i];
  }

// Iterate all subgrids
#pragma omp parallel for schedule(guided)
  for (int s = 0; s < nr_subgrids; s++) {
    // Load metadata
    const idg::Metadata m = metadata[s];
    const int time_offset = m.time_index;
    const int nr_timesteps = m.nr_timesteps;
    const int channel_begin = m.channel_begin;
    const int channel_end = m.channel_end;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
    const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

    // Initialize aterm index to first timestep
    unsigned int aterm_idx_previous = aterm_indices[time_offset];

    // Allocate memory
    float* pixels_xx_real = nullptr;
    float* pixels_xx_imag = nullptr;
    float* pixels_xy_real = nullptr;
    float* pixels_xy_imag = nullptr;
    float* pixels_yx_real = nullptr;
    float* pixels_yx_imag = nullptr;
    float* pixels_yy_real = nullptr;
    float* pixels_yy_imag = nullptr;
    pixels_xx_real = allocate_memory<float>(nr_pixels);
    pixels_xx_imag = allocate_memory<float>(nr_pixels);
    if (nr_correlations == 4) {
      pixels_xy_real = allocate_memory<float>(nr_pixels);
      pixels_xy_imag = allocate_memory<float>(nr_pixels);
      pixels_yx_real = allocate_memory<float>(nr_pixels);
      pixels_yx_imag = allocate_memory<float>(nr_pixels);
    }
    pixels_yy_real = allocate_memory<float>(nr_pixels);
    pixels_yy_imag = allocate_memory<float>(nr_pixels);
    float* phasor_real = allocate_memory<float>(nr_pixels);
    float* phasor_imag = allocate_memory<float>(nr_pixels);
    float* phase = allocate_memory<float>(nr_pixels);
    float* phase_offset = allocate_memory<float>(nr_pixels);
    float* phase_index = allocate_memory<float>(nr_pixels);

    // Compute u and v offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float w_offset = 2 * M_PI * w_offset_in_lambda;

    // Compute phase offset
    for (unsigned i = 0; i < nr_pixels; i++) {
      phase_offset[i] = u_offset * l_offset[i] + v_offset * m_offset[i] +
                        w_offset * n_offset[i];
    }

    // Iterate all timesteps
    for (int time = 0; time < nr_timesteps; time++) {
      // Load UVW coordinates
      float u = uvw[time_offset + time].u;
      float v = uvw[time_offset + time].v;
      float w = uvw[time_offset + time].w;

      // Get aterm indices for current timestep
      const unsigned int aterm_idx_current = aterm_indices[time_offset + time];

// Determine whether aterm has changed
#if defined(__PPC__)  // workaround compiler bug
      unsigned int aterm_changed;
#else
      bool aterm_changed;
#endif
      aterm_changed = aterm_idx_previous != aterm_idx_current;

      // Compute phase index and apply phase shift.
      for (unsigned i = 0; i < nr_pixels; i++) {
        phase_index[i] = u * l_index[i] + v * m_index[i] + w * n_index[i];
      }

      // Apply aterm to subgrid
      if (time == 0 || aterm_changed) {
        for (unsigned i = 0; i < nr_pixels; i++) {
          int y = i / subgrid_size;
          int x = i % subgrid_size;

          // Load taper
          const float taper_ = taper[y * subgrid_size + x];

          // Compute shifted position in subgrid
          int x_src = (x + (subgrid_size / 2)) % subgrid_size;
          int y_src = (y + (subgrid_size / 2)) % subgrid_size;

          // Load pixel values and apply taper
          std::complex<float> pixels[4] __attribute__((aligned(ALIGNMENT)));
          if (nr_correlations == 4) {
            for (int pol = 0; pol < nr_polarizations; pol++) {
              size_t src_idx = index_subgrid(nr_polarizations, subgrid_size, s,
                                             pol, y_src, x_src);
              pixels[pol] = taper_ * subgrid[src_idx];
            }
          } else if (nr_correlations == 2) {
            size_t src_idx = index_subgrid(nr_polarizations, subgrid_size, s, 0,
                                           y_src, x_src);
            std::complex<float> value = taper_ * subgrid[src_idx];
            pixels[0] = value;
            pixels[3] = value;
          }

          // Apply aterm
          size_t station1_idx =
              index_aterm(subgrid_size, 4, nr_stations, aterm_idx_current,
                          station1, y, x, 0);
          size_t station2_idx =
              index_aterm(subgrid_size, 4, nr_stations, aterm_idx_current,
                          station2, y, x, 0);
          const std::complex<float>* aterm1_ptr = &aterms[station1_idx];
          const std::complex<float>* aterm2_ptr = &aterms[station2_idx];
          apply_aterm_degridder(pixels, aterm1_ptr, aterm2_ptr);

          // Store pixels
          pixels_xx_real[i] = pixels[0].real();
          pixels_xx_imag[i] = pixels[0].imag();
          if (nr_correlations == 4) {
            pixels_xy_real[i] = pixels[1].real();
            pixels_xy_imag[i] = pixels[1].imag();
            pixels_yx_real[i] = pixels[2].real();
            pixels_yx_imag[i] = pixels[2].imag();
          }
          pixels_yy_real[i] = pixels[3].real();
          pixels_yy_imag[i] = pixels[3].imag();
        }

        // Update aterm index
        aterm_idx_previous = aterm_idx_current;
      }

      // Iterate all channels
      for (int chan = channel_begin; chan < channel_end; chan++) {
        // Compute phase
        for (unsigned i = 0; i < nr_pixels; i++) {
          // Compute phase
          float wavenumber = wavenumbers[chan];
          phase[i] = (phase_index[i] * wavenumber) - phase_offset[i];
        }

        // Compute phasor
        compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);

        // Compute visibilities
        std::complex<float> sums[nr_correlations]
            __attribute__((aligned(ALIGNMENT)));

        if (nr_correlations == 4) {
          compute_reduction(nr_pixels, pixels_xx_real, pixels_xy_real,
                            pixels_yx_real, pixels_yy_real, pixels_xx_imag,
                            pixels_xy_imag, pixels_yx_imag, pixels_yy_imag,
                            phasor_real, phasor_imag, sums);
        } else {
          compute_reduction(nr_pixels, pixels_xx_real, pixels_yy_real,
                            pixels_xx_imag, pixels_yy_imag, phasor_real,
                            phasor_imag, sums);
        }

        // Store visibilities
        const float scale = 1.0f / nr_pixels;
        int time_idx = time_offset + time;
        int chan_idx = chan;
        size_t dst_idx = index_visibility(nr_correlations, nr_channels,
                                          time_idx, chan_idx, 0);
        for (int pol = 0; pol < nr_correlations; pol++) {
          visibilities[dst_idx + pol] = {scale * sums[pol].real(),
                                         scale * sums[pol].imag()};
        }
      }  // end for channel
    }    // end for time

    // Free memory
    free(pixels_xx_real);
    free(pixels_xy_real);
    free(pixels_yx_real);
    free(pixels_yy_real);
    free(pixels_xx_imag);
    free(pixels_xy_imag);
    free(pixels_yx_imag);
    free(pixels_yy_imag);
    free(phase);
    free(phase_offset);
    free(phase_index);
    free(phasor_real);
    free(phasor_imag);
  }  // end s
}  // end kernel_degridder

}  // end namespace optimized
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg