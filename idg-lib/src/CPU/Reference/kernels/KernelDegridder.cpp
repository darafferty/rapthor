// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>
#include <cmath>
#include <cstring>

#include "common/Types.h"
#include "common/Math.h"

namespace idg {
namespace kernel {
namespace cpu {
namespace reference {

void kernel_degridder(
    const int nr_subgrids, const int nr_polarizations, const long grid_size,
    const int subgrid_size, const float image_size,
    const float w_step_in_lambda, const float* __restrict__ shift,
    const int nr_correlations, const int nr_channels, const int nr_stations,
    const idg::UVW<float>* uvw, const float* wavenumbers,
    std::complex<float>* visibilities, const float* spheroidal,
    const std::complex<float>* aterms, const int* aterms_indices,
    const idg::Metadata* metadata, const std::complex<float>* subgrid) {
// Iterate all subgrids
#pragma omp parallel for
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

    // Compute u and v offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float w_offset = 2 * M_PI * w_offset_in_lambda;

    // Storage
    std::complex<float> pixels[subgrid_size][subgrid_size][4];

    // Iterate all timesteps
    for (int time = 0; time < nr_timesteps; time++) {
      // Reset pixels
      memset((void*)pixels, 0,
             subgrid_size * subgrid_size * 4 * sizeof(std::complex<float>));

      // Apply aterm to subgrid
      for (int y = 0; y < subgrid_size; y++) {
        for (int x = 0; x < subgrid_size; x++) {
          // Load spheroidal
          float sph = spheroidal[y * subgrid_size + x];

          // Compute shifted position in subgrid
          int x_src = (x + (subgrid_size / 2)) % subgrid_size;
          int y_src = (y + (subgrid_size / 2)) % subgrid_size;

          // Load pixel values and apply spheroidal
          if (nr_correlations == 4) {
            for (int pol = 0; pol < nr_polarizations; pol++) {
              size_t index =
                  s * nr_polarizations * subgrid_size * subgrid_size +
                  pol * subgrid_size * subgrid_size + y_src * subgrid_size +
                  x_src;
              pixels[y][x][pol] = sph * subgrid[index];
            }
          } else if (nr_correlations == 2) {
            size_t index = s * nr_polarizations * subgrid_size * subgrid_size +
                           y_src * subgrid_size + x_src;
            pixels[y][x][0] = sph * subgrid[index];
            pixels[y][x][3] = sph * subgrid[index];
          }

          // Load aterm index
          int aterm_index = aterms_indices[time_offset + time];

          // Load aterm for station1
          int station1_index = (aterm_index * nr_stations + station1) *
                                   subgrid_size * subgrid_size * 4 +
                               y * subgrid_size * 4 + x * 4;
          const std::complex<float>* aterms1 =
              (std::complex<float>*)&aterms[station1_index];

          // Load aterm for station2
          int station2_index = (aterm_index * nr_stations + station2) *
                                   subgrid_size * subgrid_size * 4 +
                               y * subgrid_size * 4 + x * 4;
          const std::complex<float>* aterms2 =
              (std::complex<float>*)&aterms[station2_index];

          apply_aterm_degridder(pixels[y][x], aterms1, aterms2);
        }  // end for x
      }    // end for y

      // Load UVW coordinates
      float u = uvw[time_offset + time].u;
      float v = uvw[time_offset + time].v;
      float w = uvw[time_offset + time].w;

      // Iterate all channels
      for (int chan = channel_begin; chan < channel_end; chan++) {
        // Update all polarizations
        std::complex<float> sum[nr_correlations];
        memset((void*)sum, 0, nr_correlations * sizeof(std::complex<float>));

        // Iterate all pixels in subgrid
        for (int y = 0; y < subgrid_size; y++) {
          for (int x = 0; x < subgrid_size; x++) {
            // Compute l,m,n for phase offset and phase index calculation.
            const float l_offset = compute_l(x, subgrid_size, image_size);
            const float m_offset = compute_m(y, subgrid_size, image_size);
            const float l_index = l_offset + shift[0];  // l: Positive direction
            const float m_index = m_offset - shift[1];  // m: Negative direction
            const float n = compute_n(l_index, m_index);

            // Compute phase offset
            const float phase_offset =
                u_offset * l_offset + v_offset * m_offset + w_offset * n;

            // Compute phase index, including phase shift.
            const float phase_index = u * l_index + v * m_index + w * n;

            // Compute phase
            const float phase =
                (phase_index * wavenumbers[chan]) - phase_offset;

            // Compute phasor
            const std::complex<float> phasor = {cosf(phase), sinf(phase)};

            // Update visibility
            if (nr_correlations == 4) {
              for (int pol = 0; pol < nr_correlations; pol++) {
                sum[pol] += pixels[y][x][pol] * phasor;
              }
            } else if (nr_correlations == 2) {
              sum[0] += pixels[y][x][0] * phasor;
              sum[1] += pixels[y][x][3] * phasor;
            }
          }  // end for x
        }    // end for y

        // Set visibility value
        const float scale = 1.0f / (subgrid_size * subgrid_size);
        size_t index = (time_offset + time) * nr_channels + chan;
        for (int pol = 0; pol < nr_correlations; pol++) {
          visibilities[index * nr_correlations + pol] = sum[pol] * scale;
        }
      }  // end for channel
    }    // end for time
  }      // end for s
}  // end kernel_degridder

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg