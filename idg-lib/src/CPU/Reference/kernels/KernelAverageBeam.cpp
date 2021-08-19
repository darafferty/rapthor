// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>
#include <cmath>
#include <vector>

#include "common/Types.h"

namespace idg {
namespace kernel {
namespace cpu {
namespace reference {

void kernel_average_beam(const unsigned int nr_baselines,
                         const unsigned int nr_antennas,
                         const unsigned int nr_timesteps,
                         const unsigned int nr_channels,
                         const unsigned int nr_aterms,
                         const unsigned int subgrid_size,
                         const unsigned int nr_polarizations,
                         const idg::UVW<float> *__restrict__ uvw_,
                         const idg::Baseline *__restrict__ baselines_,
                         const std::complex<float> *__restrict__ aterms_,
                         const unsigned int *__restrict__ aterms_offsets_,
                         const float *__restrict__ weights_,
                         std::complex<float> *__restrict__ average_beam_) {
  // Define multidimensional types
  typedef std::complex<float> AverageBeam[subgrid_size * subgrid_size]
                                         [nr_polarizations][nr_polarizations];
  typedef std::complex<float> ATerms[nr_aterms][nr_antennas][subgrid_size]
                                    [subgrid_size][nr_polarizations];
  typedef unsigned int ATermOffsets[nr_aterms + 1];
  typedef unsigned int StationPairs[nr_baselines][2];
  typedef float UVW[nr_baselines][nr_timesteps][3];
  typedef float Weights[nr_baselines][nr_timesteps][nr_channels]
                       [nr_polarizations];
  typedef float SumOfWeights[nr_baselines][nr_aterms][nr_polarizations];

  // Cast class members to multidimensional types used in this method
  const ATerms &aterms = *reinterpret_cast<const ATerms *>(aterms_);
  AverageBeam &average_beam = *reinterpret_cast<AverageBeam *>(average_beam_);
  const ATermOffsets &aterm_offsets =
      *reinterpret_cast<const ATermOffsets *>(aterms_offsets_);
  const StationPairs &station_pairs =
      *reinterpret_cast<const StationPairs *>(baselines_);
  const UVW &uvw = *reinterpret_cast<const UVW *>(uvw_);
  const Weights &weights = *reinterpret_cast<const Weights *>(weights_);

  // Initialize sum of weights
  std::vector<float> sum_of_weights_buffer(
      nr_baselines * nr_aterms * nr_polarizations, 0.0);
  SumOfWeights &sum_of_weights =
      *((SumOfWeights *)sum_of_weights_buffer.data());

// Compute sum of weights
#pragma omp parallel for
  for (unsigned int n = 0; n < nr_aterms; n++) {
    unsigned int time_start = aterm_offsets[n];
    unsigned int time_end = aterm_offsets[n + 1];

    // Loop over baselines
    for (unsigned int bl = 0; bl < nr_baselines; bl++) {
      for (unsigned int t = time_start; t < time_end; t++) {
        if (std::isinf(uvw[bl][t][0])) continue;

        for (unsigned int ch = 0; ch < nr_channels; ch++) {
          for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
            sum_of_weights[bl][n][pol] += weights[bl][t][ch][pol];
          }
        }
      }
    }
  }

// Compute average beam for all pixels
#pragma omp parallel for
  for (unsigned int i = 0; i < subgrid_size * subgrid_size; i++) {
    std::complex<double> sum[nr_polarizations][nr_polarizations];

    // Loop over aterms
    for (unsigned int n = 0; n < nr_aterms; n++) {
      // Loop over baselines
      for (unsigned int bl = 0; bl < nr_baselines; bl++) {
        unsigned int antenna1 = station_pairs[bl][0];
        unsigned int antenna2 = station_pairs[bl][1];

        // Check whether stationPair is initialized
        if (antenna1 >= nr_antennas || antenna2 >= nr_antennas) {
          continue;
        }

        std::complex<float> aXX1 = aterms[n][antenna1][0][i][0];
        std::complex<float> aXY1 = aterms[n][antenna1][0][i][1];
        std::complex<float> aYX1 = aterms[n][antenna1][0][i][2];
        std::complex<float> aYY1 = aterms[n][antenna1][0][i][3];

        std::complex<float> aXX2 = std::conj(aterms[n][antenna2][0][i][0]);
        std::complex<float> aXY2 = std::conj(aterms[n][antenna2][0][i][1]);
        std::complex<float> aYX2 = std::conj(aterms[n][antenna2][0][i][2]);
        std::complex<float> aYY2 = std::conj(aterms[n][antenna2][0][i][3]);

        std::complex<float> kp[16] = {};
        kp[0 + 0] = aXX2 * aXX1;
        kp[0 + 4] = aXX2 * aXY1;
        kp[0 + 8] = aXY2 * aXX1;
        kp[0 + 12] = aXY2 * aXY1;

        kp[1 + 0] = aXX2 * aYX1;
        kp[1 + 4] = aXX2 * aYY1;
        kp[1 + 8] = aXY2 * aYX1;
        kp[1 + 12] = aXY2 * aYY1;

        kp[2 + 0] = aYX2 * aXX1;
        kp[2 + 4] = aYX2 * aXY1;
        kp[2 + 8] = aYY2 * aXX1;
        kp[2 + 12] = aYY2 * aXY1;

        kp[3 + 0] = aYX2 * aYX1;
        kp[3 + 4] = aYX2 * aYY1;
        kp[3 + 8] = aYY2 * aYX1;
        kp[3 + 12] = aYY2 * aYY1;

        for (unsigned int ii = 0; ii < nr_polarizations; ii++) {
          for (unsigned int jj = 0; jj < nr_polarizations; jj++) {
            // Load weights for current baseline, aterm
            float *weights = &sum_of_weights[bl][n][0];

            // Compute real and imaginary part of update separately
            float update_real = 0;
            float update_imag = 0;
            for (unsigned int p = 0; p < nr_polarizations; p++) {
              float kp1_real = kp[4 * ii + p].real();
              float kp1_imag = -kp[4 * ii + p].imag();
              float kp2_real = kp[4 * jj + p].real();
              float kp2_imag = kp[4 * jj + p].imag();
              update_real +=
                  weights[p] * (kp1_real * kp2_real - kp1_imag * kp2_imag);
              update_imag +=
                  weights[p] * (kp1_real * kp2_imag + kp1_imag * kp2_real);
            }

            // Add kronecker product to sum
            sum[ii][jj] += std::complex<float>(update_real, update_imag);
          }
        }
      }  // end for baselines
    }    // end for aterms

    // Set average beam from sum of kronecker products
    for (size_t ii = 0; ii < 4; ii++) {
      for (size_t jj = 0; jj < 4; jj++) {
        average_beam[i][ii][jj] += sum[ii][jj];
      }
    }
  }  // end for pixels
}  // end kernel_average_beam

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg