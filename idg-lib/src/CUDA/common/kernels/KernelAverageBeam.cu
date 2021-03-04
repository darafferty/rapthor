// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "math.cu"
#include "Types.h"

#define MAX_NR_ATERMS 64

inline __device__ size_t index_weight(
    unsigned int nr_time,
    unsigned int nr_channels,
    unsigned int bl,
    unsigned int time,
    unsigned int chan,
    unsigned int pol) {
  // weights: [nr_baselines][nr_time][nr_channels][NR_CORRELATIONS]
  return bl * nr_time * nr_channels * NR_CORRELATIONS +
                 time * nr_channels * NR_CORRELATIONS +
                               chan * NR_CORRELATIONS +
                                                    pol;
}

inline __device__ size_t index_average_beam(
    unsigned int i,
    unsigned int ii,
    unsigned int jj) {
  // average_beam: [subgrid_size*subgrid_size][4][4]
  return i * 4 * 4 +
            ii * 4 +
                  jj;
}

/*
    Kernel
*/
extern "C" {
__global__ void kernel_average_beam(
    const unsigned int                  nr_antennas,
    const unsigned int                  nr_timesteps,
    const unsigned int                  nr_channels,
    const unsigned int                  nr_aterms,
    const unsigned int                  subgrid_size,
    const UVW<float>*      __restrict__ uvw,
    const Baseline*        __restrict__ baselines,
    const float2*          __restrict__ aterms,
    const int*             __restrict__ aterms_offsets,
    const float*           __restrict__ weights,
         double2*          __restrict__ average_beam)
{
  unsigned int bl = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_threads = blockDim.x;

  unsigned int antenna1 = baselines[bl].station1;
  unsigned int antenna2 = baselines[bl].station2;

  // Check whether stationPair is initialized
  if (antenna1 >= nr_antennas || antenna2 >= nr_antennas) {
    return;
  }

  int nr_aterm_blocks = (nr_aterms+MAX_NR_ATERMS-1)/MAX_NR_ATERMS;
  for(int aterm_block_idx = 0; aterm_block_idx < nr_aterm_blocks; ++aterm_block_idx) {
    float sum_of_weights[MAX_NR_ATERMS][NR_CORRELATIONS];
    memset(sum_of_weights, 0, MAX_NR_ATERMS * NR_CORRELATIONS * sizeof(float));

    // Compute average beam for all pixels
    for (unsigned int i = tid; i < (subgrid_size * subgrid_size); i += num_threads) {
      unsigned y = i / subgrid_size;
      unsigned x = i % subgrid_size;

      double2 sum[NR_CORRELATIONS][NR_CORRELATIONS];
      memset(sum, 0, NR_CORRELATIONS * NR_CORRELATIONS * sizeof(double2));

      // Loop over aterms
      int nr_aterms_current_block = min(MAX_NR_ATERMS, nr_aterms - aterm_block_idx*MAX_NR_ATERMS);
      for (unsigned int n = 0; n < nr_aterms_current_block; n++) {

        // Compute sum of weights
        if (i == tid) {
          unsigned int time_start = aterms_offsets[n + aterm_block_idx*MAX_NR_ATERMS];
          unsigned int time_end = aterms_offsets[n + 1 + aterm_block_idx*MAX_NR_ATERMS];

          for (unsigned int t = time_start; t < time_end; t++) {
            float u = uvw[bl * nr_timesteps + t].u;
            if (isinf(u)) continue;

            for (unsigned int ch = 0; ch < nr_channels; ch++) {
              for (unsigned int pol = 0; pol < NR_CORRELATIONS; pol++) {
                  unsigned int weight_idx = index_weight(nr_timesteps, nr_channels, bl, t, ch, pol);
                  sum_of_weights[n][pol] += weights[weight_idx];
              }
            }
          } // end for time
        }

        int station1_idx = index_aterm(subgrid_size, nr_antennas, n + aterm_block_idx*MAX_NR_ATERMS, antenna1, y, x, 0);
        int station2_idx = index_aterm(subgrid_size, nr_antennas, n + aterm_block_idx*MAX_NR_ATERMS, antenna2, y, x, 0);

        float2 aXX1 = aterms[station1_idx + 0];
        float2 aXY1 = aterms[station1_idx + 1];
        float2 aYX1 = aterms[station1_idx + 2];
        float2 aYY1 = aterms[station1_idx + 3];
        float2 aXX2 = conj(aterms[station2_idx + 0]);
        float2 aXY2 = conj(aterms[station2_idx + 1]);
        float2 aYX2 = conj(aterms[station2_idx + 2]);
        float2 aYY2 = conj(aterms[station2_idx + 3]);

        float2 kp[16] = {};
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

        for (int ii = 0; ii < NR_CORRELATIONS; ii++) {
          for (int jj = 0; jj < NR_CORRELATIONS; jj++) {
            // Compute real and imaginary part of update separately
            float update_real = 0;
            float update_imag = 0;
            for (int p = 0; p < NR_CORRELATIONS; p++) {
              float weight = sum_of_weights[n][p];
              float kp1_real = kp[4 * ii + p].x;
              float kp1_imag = -kp[4 * ii + p].y;
              float kp2_real = kp[4 * jj + p].x;
              float kp2_imag = kp[4 * jj + p].y;
              update_real +=
                  weight * (kp1_real * kp2_real - kp1_imag * kp2_imag);
              update_imag +=
                  weight * (kp1_real * kp2_imag + kp1_imag * kp2_real);
            }

            // Add kronecker product to sum
            sum[ii][jj] += make_double2(update_real, update_imag);
          }
        }
      } // end for aterms

      // Set average beam from sum of kronecker products
      for (int ii = 0; ii < NR_CORRELATIONS; ii++) {
        for (int jj = 0; jj < NR_CORRELATIONS; jj++) {
          unsigned average_beam_idx = index_average_beam(i, ii, jj);
          atomicAdd(average_beam[average_beam_idx], sum[ii][jj]);
        }
      }
    } // end for pixels
  } // end for aterm_block_idx
}

} // end extern "C"