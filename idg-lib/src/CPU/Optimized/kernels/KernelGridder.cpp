// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/memory.h"
#include "common/Types.h"
#include "common/Index.h"

#include "Math.h"

inline void update_subgrid(int nr_polarizations, int nr_pixels, int nr_stations,
                           int subgrid_size, int subgrid, int aterm_index,
                           int station1, int station2, const float* spheroidal,
                           const std::complex<float>* aterms,
                           const std::complex<float>* avg_aterm_correction,
                           const std::complex<float>* subgrid_local,
                           std::complex<float>* subgrid_global) {
  // Iterate all pixels in subgrid
  for (int i = 0; i < nr_pixels; i++) {
    int y = i / subgrid_size;
    int x = i % subgrid_size;

    // Apply the conjugate transpose of the A-term
    size_t station1_idx = index_aterm(subgrid_size, 4, nr_stations, aterm_index,
                                      station1, y, x, 0);
    size_t station2_idx = index_aterm(subgrid_size, 4, nr_stations, aterm_index,
                                      station2, y, x, 0);
    const std::complex<float>* aterm1_ptr = &aterms[station1_idx];
    const std::complex<float>* aterm2_ptr = &aterms[station2_idx];
    std::complex<float> pixels[4];
    for (int pol = 0; pol < 4; pol++) {
      pixels[pol] = subgrid_local[pol * nr_pixels + i];
    }
    apply_aterm_gridder(pixels, aterm1_ptr, aterm2_ptr);

    if (avg_aterm_correction)
      apply_avg_aterm_correction(
          avg_aterm_correction + (y * subgrid_size + x) * 16, pixels);

    // Compute shifted position in subgrid
    int x_dst = (x + (subgrid_size / 2)) % subgrid_size;
    int y_dst = (y + (subgrid_size / 2)) % subgrid_size;

    // Load spheroidal
    float sph = spheroidal[y * subgrid_size + x];

    // Update global subgrid
    if (nr_polarizations == 4) {
      for (int pol = 0; pol < nr_polarizations; pol++) {
        size_t dst_idx = index_subgrid(nr_polarizations, subgrid_size, subgrid,
                                       pol, y_dst, x_dst);
        subgrid_global[dst_idx] += pixels[pol] * sph;
      }
    } else if (nr_polarizations == 1) {
      size_t dst_idx = index_subgrid(nr_polarizations, subgrid_size, subgrid, 0,
                                     y_dst, x_dst);
      subgrid_global[dst_idx] += (pixels[0] + pixels[3]) * sph * 0.5f;
    }
  }
}

namespace idg {
namespace kernel {
namespace cpu {
namespace optimized {

void kernel_gridder(const int nr_subgrids, const int nr_polarizations,
                    const long grid_size, const int subgrid_size,
                    const float image_size, const float w_step_in_lambda,
                    const float* __restrict__ shift, const int nr_correlations,
                    const int nr_channels, const int nr_stations,
                    const idg::UVW<float>* uvw, const float* wavenumbers,
                    const std::complex<float>* visibilities,
                    const float* spheroidal, const std::complex<float>* aterms,
                    const int* aterms_indices,
                    const std::complex<float>* avg_aterm_correction,
                    const idg::Metadata* metadata,
                    std::complex<float>* subgrid) {
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
    // Initialize global subgrid
    size_t subgrid_idx =
        index_subgrid(nr_polarizations, subgrid_size, s, 0, 0, 0);
    std::complex<float>* subgrid_ptr = &subgrid[subgrid_idx];
    memset(static_cast<void*>(subgrid_ptr), 0,
           nr_polarizations * nr_pixels * sizeof(std::complex<float>));

    // Load metadata
    const idg::Metadata m = metadata[s];
    const int time_offset_global = m.time_index;
    const int nr_timesteps = m.nr_timesteps;
    const int channel_begin = m.channel_begin;
    const int channel_end = m.channel_end;
    const int nr_channels_subgrid = channel_end - channel_begin;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
    const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

    // Allocate memory
    size_t total_nr_visibilities = nr_timesteps * nr_channels_subgrid;
    float* vis_xx_real = nullptr;
    float* vis_xx_imag = nullptr;
    float* vis_xy_real = nullptr;
    float* vis_xy_imag = nullptr;
    float* vis_yx_real = nullptr;
    float* vis_yx_imag = nullptr;
    float* vis_yy_real = nullptr;
    float* vis_yy_imag = nullptr;

    vis_xx_real = allocate_memory<float>(total_nr_visibilities);
    vis_xx_imag = allocate_memory<float>(total_nr_visibilities);
    if (nr_correlations == 4) {
      vis_xy_real = allocate_memory<float>(total_nr_visibilities);
      vis_xy_imag = allocate_memory<float>(total_nr_visibilities);
      vis_yx_real = allocate_memory<float>(total_nr_visibilities);
      vis_yx_imag = allocate_memory<float>(total_nr_visibilities);
    }
    vis_yy_real = allocate_memory<float>(total_nr_visibilities);
    vis_yy_imag = allocate_memory<float>(total_nr_visibilities);
    float* phasor_real = allocate_memory<float>(total_nr_visibilities);
    float* phasor_imag = allocate_memory<float>(total_nr_visibilities);
    float* phase = allocate_memory<float>(total_nr_visibilities);

    // Initialize local subgrid
    //  - NR_CORRELATIONS=4, all four polarizations are used.
    //  - NR_CORRELATIONS=2, use only first and last polarization index.
    std::complex<float>* subgrid_local =
        allocate_memory<std::complex<float>>(4 * nr_pixels);
    memset(static_cast<void*>(subgrid_local), 0,
           4 * nr_pixels * sizeof(std::complex<float>));

    // Initialize aterm index to first timestep
    int aterm_idx_previous = aterms_indices[time_offset_global];

    // Compute u and v offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float w_offset = 2 * M_PI * w_offset_in_lambda;

    // Iterate all timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps;
         time_offset_local += current_nr_timesteps) {
      // Get aterm indices for current timestep
      int time_current = time_offset_global + time_offset_local;
      int aterm_idx_current = aterms_indices[time_current];

// Determine whether aterm has changed
#if defined(__PPC__)  // workaround compiler bug
      unsigned int aterm_changed;
#else
      bool aterm_changed;
#endif
      aterm_changed = aterm_idx_previous != aterm_idx_current;

      // Determine number of timesteps to process
      current_nr_timesteps = 0;
      for (int time = time_offset_local; time < nr_timesteps; time++) {
        if (aterms_indices[time_offset_global + time] == aterm_idx_current) {
          current_nr_timesteps++;
        } else {
          break;
        }
      }

      if (aterm_changed) {
        // Update subgrid
        update_subgrid(nr_polarizations, nr_pixels, nr_stations, subgrid_size,
                       s, aterm_idx_previous, station1, station2, spheroidal,
                       aterms, avg_aterm_correction, subgrid_local, subgrid);

        // Reset local subgrid for new aterms
        memset(static_cast<void*>(subgrid_local), 0,
               4 * nr_pixels * sizeof(std::complex<float>));

        // Update aterm indices
        aterm_idx_previous = aterm_idx_current;
      }

      // Load visibilities
      for (int time = 0; time < current_nr_timesteps; time++) {
        for (int chan = channel_begin; chan < channel_end; chan++) {
          int time_idx = time_offset_global + time_offset_local + time;
          int chan_idx = chan - channel_begin;
          size_t src_idx =
              index_visibility(nr_correlations, nr_channels, time_idx, chan, 0);
#if !defined(USE_EXTRAPOLATE)
          size_t dst_idx = time * nr_channels_subgrid + chan_idx;
#else
          size_t dst_idx = chan_idx * current_nr_timesteps + time;
#endif

          if (nr_correlations == 4) {
            vis_xx_real[dst_idx] = visibilities[src_idx + 0].real();
            vis_xx_imag[dst_idx] = visibilities[src_idx + 0].imag();
            vis_xy_real[dst_idx] = visibilities[src_idx + 1].real();
            vis_xy_imag[dst_idx] = visibilities[src_idx + 1].imag();
            vis_yx_real[dst_idx] = visibilities[src_idx + 2].real();
            vis_yx_imag[dst_idx] = visibilities[src_idx + 2].imag();
            vis_yy_real[dst_idx] = visibilities[src_idx + 3].real();
            vis_yy_imag[dst_idx] = visibilities[src_idx + 3].imag();
          } else if (nr_correlations == 2) {
            vis_xx_real[dst_idx] = visibilities[src_idx + 0].real();
            vis_xx_imag[dst_idx] = visibilities[src_idx + 0].imag();
            vis_yy_real[dst_idx] = visibilities[src_idx + 1].real();
            vis_yy_imag[dst_idx] = visibilities[src_idx + 1].imag();
          }
        }
      }

      // Iterate all pixels in subgrid
      for (unsigned i = 0; i < nr_pixels; i++) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;

#if !defined(USE_EXTRAPOLATE)
        // Compute phase
        float phase[current_nr_timesteps * nr_channels_subgrid]
            __attribute__((aligned(ALIGNMENT)));

        // Compute phase offset
        const float phase_offset = u_offset * l_offset[i] +
                                   v_offset * m_offset[i] +
                                   w_offset * n_offset[i];

        for (int time = 0; time < current_nr_timesteps; time++) {
          // Load UVW coordinate
          const int time_idx = time_offset_global + time_offset_local + time;
          const float u = uvw[time_idx].u;
          const float v = uvw[time_idx].v;
          const float w = uvw[time_idx].w;

          // Compute phase index, including phase shift.
          const float phase_index =
              u * l_index[i] + v * m_index[i] + w * n_index[i];

          // Compute phase
          for (int chan = channel_begin; chan < channel_end; chan++) {
            const int chan_idx = chan - channel_begin;
            const float wavenumber = wavenumbers[chan];
            phase[time * nr_channels_subgrid + chan_idx] =
                phase_offset - (phase_index * wavenumber);
          }
        }  // end time

        size_t current_nr_visibilities =
            current_nr_timesteps * nr_channels_subgrid;

        // Compute phasor
        compute_sincos(current_nr_visibilities, phase, phasor_imag,
                       phasor_real);
#else
        float phase_0_[current_nr_timesteps]
            __attribute__((aligned(ALIGNMENT)));
        float phase_d_[current_nr_timesteps]
            __attribute__((aligned(ALIGNMENT)));
        float phasor_c_real_[current_nr_timesteps]
            __attribute__((aligned(ALIGNMENT)));
        float phasor_c_imag_[current_nr_timesteps]
            __attribute__((aligned(ALIGNMENT)));
        float phasor_d_real_[current_nr_timesteps]
            __attribute__((aligned(ALIGNMENT)));
        float phasor_d_imag_[current_nr_timesteps]
            __attribute__((aligned(ALIGNMENT)));

        for (int time = 0; time < current_nr_timesteps; time++) {
          // Load UVW coordinate
          const int time_idx = time_offset_global + time_offset_local + time;
          const float u = uvw[time_idx].u;
          const float v = uvw[time_idx].v;
          const float w = uvw[time_idx].w;

          // Compute phase index and apply phase shift.
          const float phase_index =
              u * l_index[i] + v * m_index[i] + w * n_index[i];

          // Compute phase offset
          const float phase_offset = u_offset * l_offset[i] +
                                     v_offset * m_offset[i] +
                                     w_offset * n_offset[i];

          // Compute phases
          const float phase_0 = phase_offset - (phase_index * wavenumbers[0]);
          const float phase_1 =
              phase_offset - (phase_index * wavenumbers[channel_end - 1]);
          const float phase_d = (phase_1 - phase_0) / (nr_channels_subgrid - 1);
          phase_0_[time] = phase_0;
          phase_d_[time] = phase_d;
        }

        // Compute base and delta phasors
        compute_sincos(current_nr_timesteps, phase_0_, phasor_c_imag_,
                       phasor_c_real_);
        compute_sincos(current_nr_timesteps, phase_d_, phasor_d_imag_,
                       phasor_d_real_);

        // Extrapolate phasors
        compute_extrapolation(nr_channels_subgrid, current_nr_timesteps,
                              phasor_c_real_, phasor_c_imag_, phasor_d_real_,
                              phasor_d_imag_, phasor_real, phasor_imag);

        size_t current_nr_visibilities =
            current_nr_timesteps * nr_channels_subgrid;
#endif

        // Compute pixels
        std::complex<float> pixels[nr_correlations]
            __attribute__((aligned(ALIGNMENT)));
        if (nr_correlations == 4) {
          compute_reduction(current_nr_visibilities, vis_xx_real, vis_xy_real,
                            vis_yx_real, vis_yy_real, vis_xx_imag, vis_xy_imag,
                            vis_yx_imag, vis_yy_imag, phasor_real, phasor_imag,
                            pixels);
        } else if (nr_correlations == 2) {
          compute_reduction(current_nr_visibilities, vis_xx_real, vis_yy_real,
                            vis_xx_imag, vis_yy_imag, phasor_real, phasor_imag,
                            pixels);
        }

        // Update local subgrid
        if (nr_correlations == 4) {
          for (int pol = 0; pol < 4; pol++) {
            size_t idx =
                index_subgrid(nr_polarizations, subgrid_size, 0, pol, y, x);
            subgrid_local[idx] += pixels[pol];
          }
        } else if (nr_correlations == 2) {
          size_t idx_xx = index_subgrid(4, subgrid_size, 0, 0, y, x);
          size_t idx_yy = index_subgrid(4, subgrid_size, 0, 3, y, x);
          subgrid_local[idx_xx] += pixels[0];
          subgrid_local[idx_yy] += pixels[1];
        }
      }  // end for i (pixels)
    }    // end time_offset_local

    update_subgrid(nr_polarizations, nr_pixels, nr_stations, subgrid_size, s,
                   aterm_idx_previous, station1, station2, spheroidal, aterms,
                   avg_aterm_correction, subgrid_local, subgrid);

    // Free memory
    free(vis_xx_real);
    free(vis_xy_real);
    free(vis_yx_real);
    free(vis_yy_real);
    free(vis_xx_imag);
    free(vis_xy_imag);
    free(vis_yx_imag);
    free(vis_yy_imag);
    free(phase);
    free(phasor_real);
    free(phasor_imag);
    free(subgrid_local);
  }  // end s
}  // end kernel_gridder

}  // end namespace optimized
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg