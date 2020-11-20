// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>
#include <algorithm>
#include <vector>
#include <iostream>

#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "Types.h"
#include "Index.h"
#include "idg-fft.h"

extern "C" {
void kernel_splitter_subgrids_from_wtiles(
    const long nr_subgrids, const int grid_size, const int subgrid_size,
    const int wtile_size, const idg::Metadata *metadata, idg::float2 *subgrid,
    const idg::float2 *tiles) {
  // Precompute phasor
  float phasor_real[subgrid_size][subgrid_size];
  float phasor_imag[subgrid_size][subgrid_size];

#pragma omp parallel for collapse(2)
  for (int y = 0; y < subgrid_size; y++) {
    for (int x = 0; x < subgrid_size; x++) {
      float phase = -M_PI * (x + y - subgrid_size) / subgrid_size;
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

      int tile_index = metadata[s].wtile_index;
      int tile_top = metadata[s].wtile_coordinate.x * wtile_size -
                     subgrid_size / 2 + grid_size / 2;
      int tile_left = metadata[s].wtile_coordinate.y * wtile_size -
                      subgrid_size / 2 + grid_size / 2;

      // position in tile
      int subgrid_x = metadata[s].coordinate.x - tile_top;
      ;
      int subgrid_y = metadata[s].coordinate.y - tile_left;

      // iterate over subgrid rows, starting at a row that belongs to this
      // thread and stepping by the number of threads

      int start_y =
          (num_threads - (subgrid_y % num_threads) + thread_id) % num_threads;
      for (int y = start_y; y < subgrid_size; y += num_threads) {
        // Iterate all columns of subgrid
        for (int x = 0; x < subgrid_size; x++) {
          // Compute position in subgrid
          int x_src = (x + (subgrid_size / 2)) % subgrid_size;
          int y_src = (y + (subgrid_size / 2)) % subgrid_size;

          // Compute position in grid
          int x_dst = subgrid_x + x;
          int y_dst = subgrid_y + y;

          // Load phasor
          idg::float2 phasor = {phasor_real[y][x], phasor_imag[y][x]};

          // Add subgrid value to tiles
          for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            long src_idx = index_grid(wtile_size + subgrid_size, tile_index,
                                      pol, y_dst, x_dst);
            long dst_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);

            subgrid[dst_idx] = phasor * tiles[src_idx];

          }  // end for pol
        }    // end for x
      }      // end for y
    }        // end for s
  }          // end parallel
}  // end kernel_adder_subgrids_to_wtiles

int next_composite(int n) {
  n += (n & 1);
  while (true) {
    int nn = n;
    while ((nn % 2) == 0) nn /= 2;
    while ((nn % 3) == 0) nn /= 3;
    while ((nn % 5) == 0) nn /= 5;
    if (nn == 1) return n;
    n += 2;
  }
}

void kernel_splitter_wtiles_from_grid(int grid_size, int subgrid_size,
                                      int wtile_size, float image_size,
                                      float w_step, int nr_tiles, int *tile_ids,
                                      idg::Coordinate *tile_coordinates,
                                      idg::float2 *tiles, idg::float2 *grid) {
  float max_abs_w = 0.0;
  for (int i = 0; i < nr_tiles; i++) {
    idg::Coordinate &coordinate = tile_coordinates[i];
    float w = (coordinate.z + 0.5f) * w_step;
    max_abs_w = std::max(max_abs_w, std::abs(w));
  }

  int padded_tile_size = wtile_size + subgrid_size;

  int max_tile_size = next_composite(
      padded_tile_size + int(ceil(max_abs_w * image_size * image_size)));

  std::vector<idg::float2> tile_buffer(max_tile_size * max_tile_size *
                                       NR_POLARIZATIONS);

  for (int i = 0; i < nr_tiles; i++) {
    idg::Coordinate &coordinate = tile_coordinates[i];
    float w = (coordinate.z + 0.5f) * w_step;
    int w_padded_tile_size = next_composite(
        padded_tile_size + int(ceil(std::abs(w) * image_size * image_size)));
    int w_padding = w_padded_tile_size - padded_tile_size;
    int w_padding2 = w_padding / 2;
    size_t current_buffer_size =
        w_padded_tile_size * w_padded_tile_size * NR_POLARIZATIONS;

    tile_buffer.assign(current_buffer_size, {0.0, 0.0});

    int x0 = coordinate.x * wtile_size - (w_padded_tile_size - wtile_size) / 2 +
             grid_size / 2;
    int y0 = coordinate.y * wtile_size - (w_padded_tile_size - wtile_size) / 2 +
             grid_size / 2;
    int x_start = std::max(0, x0);
    int y_start = std::max(0, y0);
    int x_end = std::min(x0 + w_padded_tile_size, grid_size);
    int y_end = std::min(y0 + w_padded_tile_size, grid_size);

// split tile from grid
#pragma omp parallel for
    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        tile_buffer[(y - y0) * w_padded_tile_size + x - x0] =
            grid[index_grid(grid_size, 0, y, x)];
        tile_buffer[w_padded_tile_size * w_padded_tile_size +
                    (y - y0) * w_padded_tile_size + x - x0] =
            grid[index_grid(grid_size, 2, y, x)];
        tile_buffer[2 * w_padded_tile_size * w_padded_tile_size +
                    (y - y0) * w_padded_tile_size + x - x0] =
            grid[index_grid(grid_size, 1, y, x)];
        tile_buffer[3 * w_padded_tile_size * w_padded_tile_size +
                    (y - y0) * w_padded_tile_size + x - x0] =
            grid[index_grid(grid_size, 3, y, x)];
      }
    }

    idg::ifft2f(NR_POLARIZATIONS, w_padded_tile_size, w_padded_tile_size,
                (std::complex<float> *)tile_buffer.data());

    float cell_size = image_size / w_padded_tile_size;

    int N = w_padded_tile_size * w_padded_tile_size;
#pragma omp parallel for
    for (int y = 0; y < w_padded_tile_size; y++) {
      for (int x = 0; x < w_padded_tile_size; x++) {
        // Compute phase
        const float l = (y - (w_padded_tile_size / 2)) * cell_size;
        const float m = (x - (w_padded_tile_size / 2)) * cell_size;
        // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
        // accurately for small values of l and m
        const float tmp = (l * l) + (m * m);
        const float n = tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
        const float phase = 2 * M_PI * n * w;

        // Compute phasor
        idg::float2 phasor = {std::cos(phase) / N, std::sin(phase) / N};

        // Apply correction
        tile_buffer[y * w_padded_tile_size + x] =
            tile_buffer[y * w_padded_tile_size + x] * phasor;
        tile_buffer[w_padded_tile_size * w_padded_tile_size +
                    y * w_padded_tile_size + x] =
            tile_buffer[w_padded_tile_size * w_padded_tile_size +
                        y * w_padded_tile_size + x] *
            phasor;
        tile_buffer[2 * w_padded_tile_size * w_padded_tile_size +
                    y * w_padded_tile_size + x] =
            tile_buffer[2 * w_padded_tile_size * w_padded_tile_size +
                        y * w_padded_tile_size + x] *
            phasor;
        tile_buffer[3 * w_padded_tile_size * w_padded_tile_size +
                    y * w_padded_tile_size + x] =
            tile_buffer[3 * w_padded_tile_size * w_padded_tile_size +
                        y * w_padded_tile_size + x] *
            phasor;
      }
    }

    idg::fft2f(NR_POLARIZATIONS, w_padded_tile_size, w_padded_tile_size,
               (std::complex<float> *)tile_buffer.data());

    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
      for (int x = 0; x < padded_tile_size; x++) {
        int x2 = x + w_padding2;
        for (int y = 0; y < padded_tile_size; y++) {
          int y2 = y + w_padding2;
          tiles[tile_ids[i] * padded_tile_size * padded_tile_size *
                    NR_POLARIZATIONS +
                pol * padded_tile_size * padded_tile_size +
                x * padded_tile_size + y] =
              tile_buffer[pol * w_padded_tile_size * w_padded_tile_size +
                          x2 * w_padded_tile_size + y2];
        }
      }
    }
  }
}  // end kernel_splitter_wtiles_from_grid

}  // end extern "C"
