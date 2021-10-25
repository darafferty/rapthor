// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "common/Types.h"
#include "common/Index.h"

namespace idg {
namespace kernel {
namespace cpu {
namespace reference {

void kernel_adder(const long nr_subgrids, const int nr_polarizations,
                  const long grid_size, const int subgrid_size,
                  const idg::Metadata* metadata,
                  const std::complex<float>* subgrid,
                  std::complex<float>* grid) {
#pragma omp parallel for
  for (int pol = 0; pol < nr_polarizations; pol++) {
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

      // Determine polarization index
      const int index_pol_default[nr_polarizations] = {0, 1, 2, 3};
      const int index_pol_transposed[nr_polarizations] = {0, 2, 1, 3};
      int* index_pol =
          (int*)(negative_w ? index_pol_default : index_pol_transposed);

      // Check wheter subgrid fits in grid
      if (subgrid_x > 0 && subgrid_x < grid_size - subgrid_size &&
          subgrid_y > 0 && subgrid_y < grid_size - subgrid_size) {
        for (int y = 0; y < subgrid_size; y++) {
          for (int x = 0; x < subgrid_size; x++) {
            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size / 2)) % subgrid_size;
            int y_src = (y + (subgrid_size / 2)) % subgrid_size;

            // Compute position in grid
            int x_dst = subgrid_x + x;
            int y_dst = subgrid_y + y;

            // Compute phasor
            float phase = M_PI * (x + y - subgrid_size) / subgrid_size;
            std::complex<float> phasor = {cosf(phase), sinf(phase)};

            // Add subgrid value to grid
            int pol_dst = index_pol[pol];
            long dst_idx = index_grid_3d(grid_size, pol, y_dst, x_dst);
            long src_idx = index_subgrid(nr_polarizations, subgrid_size, s,
                                         pol_dst, y_src, x_src);
            std::complex<float> value = phasor * subgrid[src_idx];
            value = negative_w ? conj(value) : value;
            grid[dst_idx] += value;
          }
        }
      }
    }
  }
}

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg