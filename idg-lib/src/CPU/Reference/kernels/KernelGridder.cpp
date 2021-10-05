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

void kernel_gridder(
    const int nr_subgrids, const int nr_polarizations, const long grid_size,
    const int subgrid_size, const float image_size,
    const float w_step_in_lambda, const float* shift, const int nr_correlations,
    const int nr_channels, const int nr_stations, const idg::UVW<float>* uvw,
    const float* wavenumbers, const std::complex<float>* visibilities,
    const float* spheroidal, const std::complex<float>* aterms,
    const int* aterms_indices, const std::complex<float>* avg_aterm_correction,
    const idg::Metadata* metadata, std::complex<float>* subgrid) {
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
    std::complex<float> pixels[4][subgrid_size][subgrid_size];
    memset((void*)pixels, 0,
           subgrid_size * subgrid_size * 4 * sizeof(std::complex<float>));

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

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
          // Pixel
          std::complex<float> pixel[4];
          memset((void*)pixel, 0, 4 * sizeof(std::complex<float>));

          // Load UVW coordinates
          const float u = uvw[time_offset + time].u;
          const float v = uvw[time_offset + time].v;
          const float w = uvw[time_offset + time].w;

          // Compute phase index, including phase shift.
          const float phase_index = u * l_index + v * m_index + w * n;

          // Update pixel for every channel
          for (int chan = channel_begin; chan < channel_end; chan++) {
            // Compute phase
            float phase = phase_offset - (phase_index * wavenumbers[chan]);

            // Compute phasor
            std::complex<float> phasor = {cosf(phase), sinf(phase)};

            // Update pixel for every polarization
            size_t index = (time_offset + time) * nr_channels + chan;
            if (nr_correlations == 4) {
              for (int pol = 0; pol < nr_correlations; pol++) {
                std::complex<float> visibility =
                    visibilities[index * nr_correlations + pol];
                pixel[pol] += visibility * phasor;
              }
            } else if (nr_correlations == 2) {
              std::complex<float> visibility_xx =
                  visibilities[index * nr_correlations + 0];
              std::complex<float> visibility_yy =
                  visibilities[index * nr_correlations + 1];
              pixel[0] += visibility_xx * phasor;
              pixel[3] += visibility_yy * phasor;
            }
          }  // end for channel

          // Load aterm index
          int aterm_index = aterms_indices[time_offset + time];

          // Load a term for station1
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

          apply_aterm_gridder(pixel, aterms1, aterms2);

          // Update pixel
          if (nr_correlations == 4) {
            for (int pol = 0; pol < nr_polarizations; pol++) {
              pixels[pol][y][x] += pixel[pol];
            }
          } else if (nr_correlations == 2) {
            pixels[0][y][x] += pixel[0];
            pixels[3][y][x] += pixel[3];
          }
        }  // end for time

        // Load pixel
        std::complex<float> pixel[4];
        for (int pol = 0; pol < 4; pol++) {
          pixel[pol] = pixels[pol][y][x];
        }

        if (avg_aterm_correction) {
          apply_avg_aterm_correction(
              avg_aterm_correction + (y * subgrid_size + x) * 16, pixel);
        }

        // Load spheroidal
        float sph = spheroidal[y * subgrid_size + x];

        // Compute shifted position in subgrid
        int x_dst = (x + (subgrid_size / 2)) % subgrid_size;
        int y_dst = (y + (subgrid_size / 2)) % subgrid_size;

        // Set subgrid value
        if (nr_correlations == 4) {
          for (int pol = 0; pol < nr_polarizations; pol++) {
            size_t index = s * nr_polarizations * subgrid_size * subgrid_size +
                           pol * subgrid_size * subgrid_size +
                           y_dst * subgrid_size + x_dst;
            subgrid[index] = pixel[pol] * sph;
          }
        } else if (nr_correlations == 2) {
          size_t index = s * nr_polarizations * subgrid_size * subgrid_size +
                         0 * subgrid_size * subgrid_size +
                         y_dst * subgrid_size + x_dst;
          subgrid[index] = ((pixel[0] * sph) + (pixel[3] * sph)) * 0.5f;
        }
      }  // end for x
    }    // end for y
  }      // end for s
}  // end kernel_gridder

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg