// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>
#include <cmath>
#include <cstring>

#include "Types.h"
#include "Math.h"

extern "C" {
void kernel_degridder(
    const int nr_subgrids, const int gridsize, const int subgridsize,
    const float imagesize, const float w_step_in_lambda,
    const float* __restrict__ shift, const int nr_channels,
    const int nr_stations, const idg::UVW<float>* uvw, const float* wavenumbers,
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
    const float u_offset = (x_coordinate + subgridsize / 2 - gridsize / 2) *
                           (2 * M_PI / imagesize);
    const float v_offset = (y_coordinate + subgridsize / 2 - gridsize / 2) *
                           (2 * M_PI / imagesize);
    const float w_offset = 2 * M_PI * w_offset_in_lambda;

    // Storage
    std::complex<float> pixels[subgridsize][subgridsize][NR_POLARIZATIONS];

    // Iterate all timesteps
    for (int time = 0; time < nr_timesteps; time++) {
      // Reset pixels
      memset(pixels, 0,
             subgridsize * subgridsize * NR_POLARIZATIONS *
                 sizeof(std::complex<float>));

      // Apply aterm to subgrid
      for (int y = 0; y < subgridsize; y++) {
        for (int x = 0; x < subgridsize; x++) {
          // Load spheroidal
          float sph = spheroidal[y * subgridsize + x];

          // Compute shifted position in subgrid
          int x_src = (x + (subgridsize / 2)) % subgridsize;
          int y_src = (y + (subgridsize / 2)) % subgridsize;

          // Load pixel values and apply spheroidal
          for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            pixels[y][x][pol] =
                sph * subgrid[s * NR_POLARIZATIONS * subgridsize * subgridsize +
                              pol * subgridsize * subgridsize +
                              y_src * subgridsize + x_src];
          }

          // Load aterm index
          int aterm_index = aterms_indices[time_offset + time];

          // Load aterm for station1
          int station1_index =
              (aterm_index * nr_stations + station1) * subgridsize *
                  subgridsize * NR_POLARIZATIONS +
              y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
          const std::complex<float>* aterms1 =
              (std::complex<float>*)&aterms[station1_index];

          // Load aterm for station2
          int station2_index =
              (aterm_index * nr_stations + station2) * subgridsize *
                  subgridsize * NR_POLARIZATIONS +
              y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
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
        std::complex<float> sum[NR_POLARIZATIONS];
        memset(sum, 0, NR_POLARIZATIONS * sizeof(std::complex<float>));

        // Iterate all pixels in subgrid
        for (int y = 0; y < subgridsize; y++) {
          for (int x = 0; x < subgridsize; x++) {
            // Compute l,m,n
            const float l = compute_l(x, subgridsize, imagesize);
            const float m = compute_m(y, subgridsize, imagesize);
            const float n = compute_n(-l, m, shift);

            // Compute phase index
            float phase_index = u * l + v * m + w * n;

            // Compute phase offset
            float phase_offset = u_offset * l + v_offset * m + w_offset * n;

            // Compute phase
            float phase = (phase_index * wavenumbers[chan]) - phase_offset;

            // Compute phasor
            std::complex<float> phasor = {cosf(phase), sinf(phase)};

            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
              sum[pol] += pixels[y][x][pol] * phasor;
            }
          }  // end for x
        }    // end for y

        const float scale = 1.0f / (subgridsize * subgridsize);
        size_t index = (time_offset + time) * nr_channels + chan;
        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
          visibilities[index * NR_POLARIZATIONS + pol] = sum[pol] * scale;
        }
      }  // end for channel
    }    // end for time
  }      // end for s
}  // end kernel_degridder
}  // end extern "C"
