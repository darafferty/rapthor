// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>

#include <omp.h>

#include "common/Types.h"
#include "common/Index.h"

namespace idg {
namespace kernel {
namespace cpu {
namespace optimized {

void kernel_adder_wstack(const int nr_subgrids, const long grid_size,
                         const int subgrid_size, const idg::Metadata* metadata,
                         const std::complex<float>* subgrid,
                         std::complex<float>* grid) {
  // Precompute phasor
  float phasor_real[subgrid_size][subgrid_size];
  float phasor_imag[subgrid_size][subgrid_size];

#pragma omp parallel for collapse(2)
  for (int y = 0; y < subgrid_size; y++) {
    for (int x = 0; x < subgrid_size; x++) {
      float phase = M_PI * (x + y - subgrid_size) / subgrid_size;
      phasor_real[y][x] = cosf(phase);
      phasor_imag[y][x] = sinf(phase);
    }
  }

#pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();

    for (int s = 0; s < nr_subgrids; s++) {
      // Load subgrid coordinates
      int subgrid_x = metadata[s].coordinate.x;
      int subgrid_y = metadata[s].coordinate.y;
      int subgrid_w = metadata[s].coordinate.z;

      // Mirror subgrid coordinates for negative w-values
      bool negative_w = subgrid_w < 0;
      if (negative_w) {
        subgrid_x = grid_size - subgrid_x - subgrid_size + 1;
        subgrid_y = grid_size - subgrid_y - subgrid_size + 1;
        subgrid_w = -subgrid_w - 1;
      }

      // Check whether subgrid fits in grid
      if (!(subgrid_x >= 1 && subgrid_x < grid_size - subgrid_size &&
            subgrid_y >= 1 && subgrid_y < grid_size - subgrid_size))
        continue;

      // Determine polarization index
      const int index_pol_default[NR_POLARIZATIONS] = {0, 1, 2, 3};
      const int index_pol_transposed[NR_POLARIZATIONS] = {0, 2, 1, 3};
      int* index_pol =
          (int*)(negative_w ? index_pol_default : index_pol_transposed);

      // iterate over subgrid rows, starting at a row that belongs to this
      // thread and stepping by the number of threads

      int start_y =
          (num_threads - (subgrid_y % num_threads) + thread_id) % num_threads;
      for (int y = start_y; y < subgrid_size; y += num_threads) {
        int y_mirrored = subgrid_size - 1 - y;
        int y_ = negative_w ? y_mirrored : y;
        // Iterate all columns of subgrid
        for (int x = 0; x < subgrid_size; x++) {
          int x_mirrored = subgrid_size - 1 - x;

          // Compute position in subgrid
          int x_ = negative_w ? x_mirrored : x;
          int x_src = (x_ + (subgrid_size / 2)) % subgrid_size;
          int y_src = (y_ + (subgrid_size / 2)) % subgrid_size;

          // Compute position in grid
          int x_dst = subgrid_x + x;
          int y_dst = subgrid_y + y;

          // Load phasor
          std::complex<float> phasor = {phasor_real[y_][x_],
                                        phasor_imag[y_][x_]};

          // Add subgrid value to grid
          for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            int pol_dst = index_pol[pol];
            long dst_idx =
                index_grid(grid_size, subgrid_w, pol_dst, y_dst, x_dst);
            long src_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
            std::complex<float> value = phasor * subgrid[src_idx];
            value = negative_w ? conj(value) : value;
            grid[dst_idx] += value;
          }  // end for pol
        }    // end for x
      }      // end for y
    }        // end for s
  }          // end parallel
}  // end kernel_adder_wstack

}  // end namespace optimized
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg