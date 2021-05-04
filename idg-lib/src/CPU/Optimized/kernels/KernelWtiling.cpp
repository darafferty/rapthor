// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>
#include <algorithm>
#include <vector>

#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <fftw3.h>

#include "common/memory.h"
#include "common/Types.h"
#include "common/Index.h"
#include "common/ArrayTypes.h"
#include "Math.h"

extern "C" {

void kernel_apply_phasor(int w_padded_tile_size, float image_size, float w_step,
                         const float *shift, idg::Coordinate &coordinate,
                         idg::float2 *tile, int sign) {
  float cell_size = image_size / w_padded_tile_size;
  int N = w_padded_tile_size * w_padded_tile_size;
  float w = (coordinate.z + 0.5f) * w_step;

  for (int y = 0; y < w_padded_tile_size; y++) {
    for (int x = 0; x < w_padded_tile_size; x++) {
      // Inline FFT shift
      int x_ = (x + (w_padded_tile_size / 2)) % w_padded_tile_size;
      int y_ = (y + (w_padded_tile_size / 2)) % w_padded_tile_size;

      // Compute phase
      const float l = (x_ - (w_padded_tile_size / 2)) * cell_size;
      const float m = (y_ - (w_padded_tile_size / 2)) * cell_size;
      const float n = compute_n(l, -m, shift);
      const float phase = sign * 2 * M_PI * n * w;

      // Compute phasor
      idg::float2 phasor = {std::cos(phase) / N, std::sin(phase) / N};

      // Apply correction
      for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        size_t idx = index_grid(w_padded_tile_size, pol, y, x);
        tile[idx] *= phasor;
      }
    }
  }
}  // end kernel_apply_phasor


inline void kernel_tile_from_grid(int wtile_size,
                                  int w_padded_tile_size,
                                  int grid_size,
                                  idg::Coordinate& coordinate,
                                  idg::float2 *tile,
                                  const idg::float2 *grid)
{
  int x0 = coordinate.x * wtile_size -
           (w_padded_tile_size - wtile_size) / 2 + grid_size / 2;
  int y0 = coordinate.y * wtile_size -
           (w_padded_tile_size - wtile_size) / 2 + grid_size / 2;
  int x_start = std::max(0, x0);
  int y_start = std::max(0, y0);
  int x_end = std::min(x0 + w_padded_tile_size, grid_size);
  int y_end = std::min(y0 + w_padded_tile_size, grid_size);

  for (int y = y_start; y < y_end; y++) {
    for (int x = x_start; x < x_end; x++) {
      for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        size_t src_idx = index_grid(grid_size, pol, y, x);
        size_t dst_idx = index_grid(w_padded_tile_size, pol, y - y0, x - x0);
        tile[dst_idx] = grid[src_idx];
      }
    }
  }
}

void kernel_tiles_from_grid(int nr_tiles,
                            int wtile_size,
                            int w_padded_tile_size,
                            int grid_size,
                            idg::Coordinate *coordinates,
                            idg::float2 *tiles,
                            const idg::float2 *grid)
{
#pragma omp parallel for
  for (int i = 0; i < nr_tiles; i++)
  {
    size_t sizeof_w_padded_tile = NR_POLARIZATIONS * w_padded_tile_size * w_padded_tile_size;
    idg::float2 *tile = &tiles[i * sizeof_w_padded_tile];
    kernel_tile_from_grid(wtile_size, w_padded_tile_size, grid_size,
                          coordinates[i], tile, grid);
  }
}

void kernel_adder_subgrids_to_wtiles(
    const long nr_subgrids, const int grid_size, const int subgrid_size,
    const int wtile_size, const idg::Metadata *metadata,
    const idg::float2 *subgrid, idg::float2 *tiles) {
  // Precompute phasor
  int nr_pixels = subgrid_size * subgrid_size;
  float *phasor_real = allocate_memory<float>(nr_pixels);
  float *phasor_imag = allocate_memory<float>(nr_pixels);
  float *phase = allocate_memory<float>(nr_pixels);

#pragma omp parallel for collapse(2)
  for (int y = 0; y < subgrid_size; y++) {
    for (int x = 0; x < subgrid_size; x++) {
      phase[y * subgrid_size + x] =
          M_PI * (x + y - subgrid_size) / subgrid_size;
    }
  }

  compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);

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
          int idx = y * subgrid_size + x;
          idg::float2 phasor = {phasor_real[idx], phasor_imag[idx]};

          // Add subgrid value to tiles
          for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            long dst_idx = index_grid(wtile_size + subgrid_size, tile_index,
                                      pol, y_dst, x_dst);
            long src_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);

            idg::float2 value = phasor * subgrid[src_idx];
            tiles[dst_idx] += value;

          }  // end for pol
        }    // end for x
      }      // end for y
    }        // end for s
  }          // end parallel

  free(phase);
  free(phasor_real);
  free(phasor_imag);
}  // end kernel_adder_subgrids_to_wtiles

void kernel_adder_wtiles_to_grid(int grid_size, int subgrid_size,
                                 int wtile_size, float image_size, float w_step,
                                 const float *shift, int nr_tiles,
                                 int *tile_ids,
                                 idg::Coordinate *tile_coordinates,
                                 idg::float2 *tiles, idg::float2 *grid) {
  // Iterate tiles in batches
  int current_nr_tiles = omp_get_max_threads();
  for (int tile_offset = 0; tile_offset < nr_tiles;
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(current_nr_tiles, nr_tiles - tile_offset);

    // Determine the max w value for the current batch
    float max_abs_w = 0.0;
    for (int i = 0; i < current_nr_tiles; i++) {
      unsigned int tile_idx = tile_offset + i;
      idg::Coordinate &coordinate = tile_coordinates[tile_idx];
      float w = (coordinate.z + 0.5f) * w_step;
      max_abs_w = std::max(max_abs_w, std::abs(w));
    }

    // Compute w_padded_tile_size
    const float image_size_shift =
        image_size + 2 * std::max(std::abs(shift[0]), std::abs(shift[1]));
    const int padded_tile_size = wtile_size + subgrid_size;
    const int w_padded_tile_size =
        next_composite(padded_tile_size +
                       int(ceil(max_abs_w * image_size_shift * image_size)));

    // Allocate tile buffers for all threads
    idg::Array4D<idg::float2> tile_buffers(current_nr_tiles, NR_POLARIZATIONS,
                                           w_padded_tile_size,
                                           w_padded_tile_size);
    tile_buffers.zero();

    // Initialize FFT plans
    int rank = 2;
    int n[] = {w_padded_tile_size, w_padded_tile_size};
    int istride = 1;
    int ostride = istride;
    int idist = n[0] * n[1];
    int odist = idist;
    int flags = FFTW_ESTIMATE;
    fftwf_plan_with_nthreads(1);
    fftwf_plan plan_forward = fftwf_plan_many_dft(
        rank, n, NR_POLARIZATIONS, nullptr, n, istride, idist, nullptr, n,
        ostride, odist, FFTW_FORWARD, flags);
    fftwf_plan plan_backward = fftwf_plan_many_dft(
        rank, n, NR_POLARIZATIONS, nullptr, n, istride, idist, nullptr, n,
        ostride, odist, FFTW_BACKWARD, flags);

    // Process the current batch of tiles
#pragma omp parallel for
    for (int i = 0; i < current_nr_tiles; i++) {
      unsigned int tile_idx = tile_offset + i;

      // Copy tile to tile buffer
      idg::Coordinate &coordinate = tile_coordinates[tile_idx];
      int w_padding = w_padded_tile_size - padded_tile_size;
      int w_padding2 = w_padding / 2;
      const int index_pol_transposed[NR_POLARIZATIONS] = {0, 2, 1, 3};

      for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int y = 0; y < padded_tile_size; y++) {
          int y2 = y + w_padding2;
          for (int x = 0; x < padded_tile_size; x++) {
            int x2 = x + w_padding2;
            size_t idx =
                index_grid(padded_tile_size, tile_ids[tile_idx], pol, y, x);
            tile_buffers(i, index_pol_transposed[pol], y2, x2) = tiles[idx];
          }
        }
      }

      // Reset tile to zero
      std::fill(
          &tiles[index_grid(padded_tile_size, tile_ids[tile_idx], 0, 0, 0)],
          &tiles[index_grid(padded_tile_size, tile_ids[tile_idx] + 1, 0, 0, 0)],
          idg::float2({0.0, 0.0}));

      // Forward FFT
      fftwf_complex *tile_ptr =
          reinterpret_cast<fftwf_complex *>(tile_buffers.data(i, 0, 0, 0));
      fftwf_execute_dft(plan_forward, tile_ptr, tile_ptr);

      // Multiply w term
      kernel_apply_phasor(w_padded_tile_size, image_size, w_step, shift,
                          coordinate, tile_buffers.data(i, 0, 0, 0), -1);

      // Backwards FFT
      fftwf_execute_dft(plan_backward, tile_ptr, tile_ptr);
    }

    // Free FFT plans
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);

    // Add current batch of tiles to grid
#pragma omp parallel
    {
      for (int i = 0; i < current_nr_tiles; i++) {
        unsigned int tile_idx = tile_offset + i;

        idg::Coordinate &coordinate = tile_coordinates[tile_idx];
        int x0 = coordinate.x * wtile_size -
                 (w_padded_tile_size - wtile_size) / 2 + grid_size / 2;
        int y0 = coordinate.y * wtile_size -
                 (w_padded_tile_size - wtile_size) / 2 + grid_size / 2;
        int x_start = std::max(0, x0);
        int y_start = std::max(0, y0);
        int x_end = std::min(x0 + w_padded_tile_size, grid_size);
        int y_end = std::min(y0 + w_padded_tile_size, grid_size);

        // Add tile to grid
#pragma omp for collapse(2)
        for (int y = y_start; y < y_end; y++) {
          for (int x = x_start; x < x_end; x++) {
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
              grid[index_grid(grid_size, pol, y, x)] +=
                  tile_buffers(i, pol, y - y0, x - x0);
            }
          }
        }
      }
    }
  }  // end for tile_offset
}  // end kernel_adder_wtiles_to_grid

void kernel_splitter_subgrids_from_wtiles(
    const long nr_subgrids, const int grid_size, const int subgrid_size,
    const int wtile_size, const idg::Metadata *metadata, idg::float2 *subgrid,
    const idg::float2 *tiles) {
  // Precompute phasor
  int nr_pixels = subgrid_size * subgrid_size;
  float *phasor_real = allocate_memory<float>(nr_pixels);
  float *phasor_imag = allocate_memory<float>(nr_pixels);
  float *phase = allocate_memory<float>(nr_pixels);

#pragma omp parallel for collapse(2)
  for (int y = 0; y < subgrid_size; y++) {
    for (int x = 0; x < subgrid_size; x++) {
      phase[y * subgrid_size + x] =
          -M_PI * (x + y - subgrid_size) / subgrid_size;
    }
  }

  compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);

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
          int idx = y * subgrid_size + x;
          idg::float2 phasor = {phasor_real[idx], phasor_imag[idx]};

          // Split subgrid from tile
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

  free(phase);
  free(phasor_real);
  free(phasor_imag);
}  // end kernel_splitter_subgrids_from_wtiles

void kernel_splitter_wtiles_from_grid(int grid_size, int subgrid_size,
                                      int wtile_size, float image_size,
                                      float w_step, const float *shift,
                                      int nr_tiles, int *tile_ids,
                                      idg::Coordinate *tile_coordinates,
                                      idg::float2 *tiles, idg::float2 *grid) {
  // Iterate tiles in batches
  int current_nr_tiles = omp_get_max_threads();
  for (int tile_offset = 0; tile_offset < nr_tiles;
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(current_nr_tiles, nr_tiles - tile_offset);

    // Determine the max w value for the current batch
    float max_abs_w = 0.0;
    for (int i = 0; i < current_nr_tiles; i++) {
      unsigned int tile_idx = tile_offset + i;
      idg::Coordinate &coordinate = tile_coordinates[tile_idx];
      float w = (coordinate.z + 0.5f) * w_step;
      max_abs_w = std::max(max_abs_w, std::abs(w));
    }

    // Compute w_padded_tile_size
    const float image_size_shift =
        image_size + 2 * std::max(std::abs(shift[0]), std::abs(shift[1]));
    const int padded_tile_size = wtile_size + subgrid_size;
    const int w_padded_tile_size =
        next_composite(padded_tile_size +
                       int(ceil(max_abs_w * image_size_shift * image_size)));

    // Initialize FFT plans
    int rank = 2;
    int n[] = {w_padded_tile_size, w_padded_tile_size};
    int istride = 1;
    int ostride = istride;
    int idist = n[0] * n[1];
    int odist = idist;
    int flags = FFTW_ESTIMATE;
    fftwf_plan_with_nthreads(1);
    fftwf_plan plan_forward = fftwf_plan_many_dft(
        rank, n, NR_POLARIZATIONS, nullptr, n, istride, idist, nullptr, n,
        ostride, odist, FFTW_FORWARD, flags);
    fftwf_plan plan_backward = fftwf_plan_many_dft(
        rank, n, NR_POLARIZATIONS, nullptr, n, istride, idist, nullptr, n,
        ostride, odist, FFTW_BACKWARD, flags);

    // Allocate tile buffers for all threads
    idg::Array4D<idg::float2> tile_buffers(current_nr_tiles, NR_POLARIZATIONS,
                                           w_padded_tile_size,
                                           w_padded_tile_size);
    tile_buffers.zero();

    // Split tile from grid
    kernel_tiles_from_grid(current_nr_tiles, wtile_size, w_padded_tile_size,
                           grid_size, &tile_coordinates[tile_offset],
                           tile_buffers.data(), grid);

#pragma omp parallel for
    for (int i = 0; i < current_nr_tiles; i++) {
      unsigned int tile_idx = tile_offset + i;
      idg::Coordinate &coordinate = tile_coordinates[tile_idx];

      // Backwards FFT
      fftwf_complex *tile_ptr =
          reinterpret_cast<fftwf_complex *>(tile_buffers.data(i, 0, 0, 0));
      fftwf_execute_dft(plan_backward, tile_ptr, tile_ptr);

      // Multiply w term
      kernel_apply_phasor(w_padded_tile_size, image_size, w_step, shift,
                          coordinate, tile_buffers.data(i, 0, 0, 0), 1);

      // Forward FFT
      fftwf_execute_dft(plan_forward, tile_ptr, tile_ptr);

      // Copy tile
      const int index_pol_transposed[NR_POLARIZATIONS] = {0, 2, 1, 3};
      int w_padding = w_padded_tile_size - padded_tile_size;
      int w_padding2 = w_padding / 2;

      for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int y = 0; y < padded_tile_size; y++) {
          int y2 = y + w_padding2;
          for (int x = 0; x < padded_tile_size; x++) {
            int x2 = x + w_padding2;
            size_t idx = index_grid(padded_tile_size, tile_ids[tile_idx],
                                    index_pol_transposed[pol], y, x);
            tiles[idx] = tile_buffers(i, pol, y2, x2);
          }
        }
      }
    }  // end for current_nr_tiles

    // Free FFT plans
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
  }  // end for tile_offset
}  // end kernel_splitter_wtiles_from_grid

}  // end extern "C"
