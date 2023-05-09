// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <complex>
#include <algorithm>
#include <vector>
#include <map>

#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <fftw3.h>

#include <xtensor/xtensor.hpp>

#include "common/memory.h"
#include "common/Types.h"
#include "common/Index.h"
#include "common/WTiles.h"
#include "Math.h"

namespace idg {
namespace kernel {
namespace cpu {
namespace optimized {

void kernel_apply_phasor(int nr_polarizations, int w_padded_tile_size,
                         float image_size, float w_step, const float* shift,
                         const idg::Coordinate& coordinate,
                         std::complex<float>* tile, int sign) {
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
      std::complex<float> phasor = {std::cos(phase) / N, std::sin(phase) / N};

      // Apply correction
      for (int pol = 0; pol < nr_polarizations; pol++) {
        size_t idx = index_grid_3d(w_padded_tile_size, pol, y, x);
        tile[idx] *= phasor;
      }
    }
  }
}  // end kernel_apply_phasor

inline void kernel_tile_from_grid(int wtile_size, int w_padded_tile_size,
                                  int nr_polarizations, int grid_size,
                                  const idg::Coordinate& coordinate,
                                  std::complex<float>* tile,
                                  const std::complex<float>* grid) {
  int x0 = coordinate.x * wtile_size - (w_padded_tile_size - wtile_size) / 2 +
           grid_size / 2;
  int y0 = coordinate.y * wtile_size - (w_padded_tile_size - wtile_size) / 2 +
           grid_size / 2;
  int x_start = std::max(0, x0);
  int y_start = std::max(0, y0);
  int x_end = std::min(x0 + w_padded_tile_size, grid_size);
  int y_end = std::min(y0 + w_padded_tile_size, grid_size);

  for (int y = y_start; y < y_end; y++) {
    for (int x = x_start; x < x_end; x++) {
      for (int pol = 0; pol < nr_polarizations; pol++) {
        size_t src_idx = index_grid_3d(grid_size, pol, y, x);
        size_t dst_idx = index_grid_3d(w_padded_tile_size, pol, y - y0, x - x0);
        tile[dst_idx] = grid[src_idx];
      }
    }
  }
}

void kernel_tiles_from_grid(int nr_tiles, int wtile_size,
                            int w_padded_tile_size, int nr_polarizations,
                            int grid_size, const idg::Coordinate* coordinates,
                            std::complex<float>* tiles,
                            const std::complex<float>* grid) {
#pragma omp parallel for
  for (int i = 0; i < nr_tiles; i++) {
    size_t sizeof_w_padded_tile =
        nr_polarizations * w_padded_tile_size * w_padded_tile_size;
    std::complex<float>* tile = &tiles[i * sizeof_w_padded_tile];
    kernel_tile_from_grid(wtile_size, w_padded_tile_size, nr_polarizations,
                          grid_size, coordinates[i], tile, grid);
  }
}

void kernel_tiles_to_grid(int nr_tiles, int wtile_size, int w_padded_tile_size,
                          int nr_polarizations, int grid_size,
                          const idg::Coordinate* coordinates,
                          const std::complex<float>* tiles,
                          std::complex<float>* grid) {
#pragma omp parallel
  {
    for (int i = 0; i < nr_tiles; i++) {
      const idg::Coordinate& coordinate = coordinates[i];
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
          for (int pol = 0; pol < nr_polarizations; pol++) {
            size_t src_idx = index_grid_4d(nr_polarizations, w_padded_tile_size,
                                           i, pol, y - y0, x - x0);
            size_t dst_idx = index_grid_3d(grid_size, pol, y, x);
            grid[dst_idx] += tiles[src_idx];
          }
        }
      }
    }
  }
}

inline void kernel_copy_tile(int nr_polarizations, int src_tile_size,
                             int dst_tile_size,
                             const std::complex<float>* src_tile,
                             std::complex<float>* dst_tile) {
  const int index_pol_transposed[nr_polarizations] = {0, 2, 1, 3};
  int padding = dst_tile_size - src_tile_size;
  int padding2 = padding / 2;
  int copy_tile_size = std::min(src_tile_size, dst_tile_size);

  for (int pol = 0; pol < nr_polarizations; pol++) {
    for (int y = 0; y < copy_tile_size; y++) {
      for (int x = 0; x < copy_tile_size; x++) {
        int src_y = y;
        int dst_y = y;
        int src_x = x;
        int dst_x = x;

        if (padding > 0) {
          dst_y += padding2;
          dst_x += padding2;
        } else if (padding < 0) {
          src_y -= padding2;
          src_x -= padding2;
        }

        size_t src_idx = index_grid_3d(src_tile_size, pol, src_y, src_x);
        size_t dst_idx = index_grid_3d(dst_tile_size, index_pol_transposed[pol],
                                       dst_y, dst_x);
        dst_tile[dst_idx] = src_tile[src_idx];
      }
    }
  }
}

void kernel_fft_composite(fftwf_plan plan, int batch, int size,
                          std::complex<float>* data) {
  fftwf_complex* in_ptr = reinterpret_cast<fftwf_complex*>(data);
  fftwf_complex* out_ptr = reinterpret_cast<fftwf_complex*>(data);

  for (int i = 0; i < batch; i++) {
    // FFT over rows
    for (int y = 0; y < size; y++) {
      uint64_t offset =
          size_t(i) * size_t(size) * size_t(size) + y * size_t(size);
      fftwf_execute_dft(plan, in_ptr + offset, out_ptr + offset);
    }

    // Iterate all columns
    for (int x = 0; x < size; x++) {
      std::complex<float> tmp[size];

      // Copy column into temporary buffer
      for (int y = 0; y < size; y++) {
        uint64_t offset =
            size_t(i) * size_t(size) * size_t(size) + y * size_t(size) + x;
        tmp[y] = data[offset];
      }

      // FFT column
      fftwf_complex* tmp_ptr = reinterpret_cast<fftwf_complex*>(tmp);
      fftwf_execute_dft(plan, tmp_ptr, tmp_ptr);

      // Store the result in the output buffer
      for (int y = 0; y < size; y++) {
        uint64_t offset =
            size_t(i) * size_t(size) * size_t(size) + y * size_t(size) + x;
        data[offset] = tmp[y];
      }
    }
  }
}

void kernel_adder_subgrids_to_wtiles(
    const long nr_subgrids, const int nr_polarizations, const int grid_size,
    const int subgrid_size, const int wtile_size, const idg::Metadata* metadata,
    const std::complex<float>* subgrid, std::complex<float>* tiles) {
  // Precompute phasor
  int nr_pixels = subgrid_size * subgrid_size;
  float* phasor_real = allocate_memory<float>(nr_pixels);
  float* phasor_imag = allocate_memory<float>(nr_pixels);
  float* phase = allocate_memory<float>(nr_pixels);

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
          std::complex<float> phasor = {phasor_real[idx], phasor_imag[idx]};

          // Add subgrid value to tiles
          for (int pol = 0; pol < nr_polarizations; pol++) {
            long dst_idx =
                index_grid_4d(nr_polarizations, wtile_size + subgrid_size,
                              tile_index, pol, y_dst, x_dst);
            long src_idx = index_subgrid(nr_polarizations, subgrid_size, s, pol,
                                         y_src, x_src);

            std::complex<float> value = phasor * subgrid[src_idx];
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

void kernel_adder_wtiles_to_grid(
    int nr_polarizations, int grid_size, int subgrid_size, int wtile_size,
    float image_size, float w_step, const float* shift, int nr_tiles,
    const int* tile_ids, const idg::Coordinate* tile_coordinates,
    std::complex<float>* tiles, std::complex<float>* grid) {
  // Compute w_padded_tile_size for all tiles
  const int padded_tile_size = wtile_size + subgrid_size;
  const float image_size_shift =
      image_size + 2 * std::max(std::abs(shift[0]), std::abs(shift[1]));
  std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
      tile_coordinates, nr_tiles, w_step, image_size, image_size_shift,
      padded_tile_size);

  // FFT plans
  typedef std::pair<fftwf_plan, fftwf_plan> fft_pair;
  std::map<size_t, fft_pair> fft_plans;

  // Iterate tiles in batches
  size_t current_nr_tiles = omp_get_max_threads();
  for (size_t tile_offset = 0; tile_offset < static_cast<size_t>(nr_tiles);
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(current_nr_tiles, nr_tiles - tile_offset);

    // Find w_padded_tile_size for current batch
    size_t w_padded_tile_size = *std::max_element(
        w_padded_tile_sizes.begin() + tile_offset,
        w_padded_tile_sizes.begin() + tile_offset + current_nr_tiles);

    // Allocate tile buffers
    xt::xtensor<std::complex<float>, 4> tile_buffers(
        {current_nr_tiles, static_cast<size_t>(nr_polarizations),
         w_padded_tile_size, w_padded_tile_size},
        std::complex<float>(0.0f, 0.0f));

    // Initialize FFT plans
    if (fft_plans.find(w_padded_tile_size) == fft_plans.end()) {
      fftwf_plan plan_forward = fftwf_plan_dft_1d(
          w_padded_tile_size, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
      fftwf_plan plan_backward = fftwf_plan_dft_1d(
          w_padded_tile_size, nullptr, nullptr, FFTW_BACKWARD, FFTW_ESTIMATE);
      fft_plans.insert({w_padded_tile_size, {plan_forward, plan_backward}});
    }

    fft_pair plan_pair = fft_plans.find(w_padded_tile_size)->second;
    fftwf_plan plan_forward = plan_pair.first;
    fftwf_plan plan_backward = plan_pair.second;

    // Process the current batch of tiles
#pragma omp parallel for
    for (size_t i = 0; i < current_nr_tiles; i++) {
      const size_t tile_idx = tile_offset + i;
      const idg::Coordinate& coordinate = tile_coordinates[tile_idx];
      std::complex<float>* tile_ptr = &tile_buffers(i, 0, 0, 0);

      // Copy tile
      size_t src_idx = index_grid_4d(nr_polarizations, padded_tile_size,
                                     tile_ids[tile_idx], 0, 0, 0);
      kernel_copy_tile(nr_polarizations, padded_tile_size, w_padded_tile_size,
                       &tiles[src_idx], tile_ptr);

      // Reset tile to zero
      std::fill(&tiles[index_grid_4d(nr_polarizations, padded_tile_size,
                                     tile_ids[tile_idx], 0, 0, 0)],
                &tiles[index_grid_4d(nr_polarizations, padded_tile_size,
                                     tile_ids[tile_idx] + 1, 0, 0, 0)],
                std::complex<float>({0.0, 0.0}));

      // Backward FFT
      kernel_fft_composite(plan_backward, nr_polarizations, w_padded_tile_size,
                           tile_ptr);

      // Multiply w term
      kernel_apply_phasor(nr_polarizations, w_padded_tile_size, image_size,
                          w_step, shift, coordinate, tile_ptr, -1);

      // Forward FFT
      kernel_fft_composite(plan_forward, nr_polarizations, w_padded_tile_size,
                           tile_ptr);
    }

    // Add current batch of tiles to grid
    kernel_tiles_to_grid(
        current_nr_tiles, wtile_size, w_padded_tile_size, nr_polarizations,
        grid_size, &tile_coordinates[tile_offset], tile_buffers.data(), grid);
  }  // end for tile_offset

  // Free FFT plans
  for (auto& entry : fft_plans) {
    fftwf_destroy_plan(entry.second.first);
    fftwf_destroy_plan(entry.second.second);
  }
}  // end kernel_adder_wtiles_to_grid

void kernel_splitter_subgrids_from_wtiles(
    const long nr_subgrids, const int nr_polarizations, const int grid_size,
    const int subgrid_size, const int wtile_size, const idg::Metadata* metadata,
    std::complex<float>* subgrid, const std::complex<float>* tiles) {
  // Precompute phasor
  int nr_pixels = subgrid_size * subgrid_size;
  float* phasor_real = allocate_memory<float>(nr_pixels);
  float* phasor_imag = allocate_memory<float>(nr_pixels);
  float* phase = allocate_memory<float>(nr_pixels);

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
          std::complex<float> phasor = {phasor_real[idx], phasor_imag[idx]};

          // Split subgrid from tile
          for (int pol = 0; pol < nr_polarizations; pol++) {
            long src_idx =
                index_grid_4d(nr_polarizations, wtile_size + subgrid_size,
                              tile_index, pol, y_dst, x_dst);
            long dst_idx = index_subgrid(nr_polarizations, subgrid_size, s, pol,
                                         y_src, x_src);

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

void kernel_splitter_wtiles_from_grid(
    int nr_polarizations, int grid_size, int subgrid_size, int wtile_size,
    float image_size, float w_step, const float* shift, int nr_tiles,
    const int* tile_ids, const idg::Coordinate* tile_coordinates,
    std::complex<float>* tiles, const std::complex<float>* grid) {
  // Compute w_padded_tile_size for all tiles
  const int padded_tile_size = wtile_size + subgrid_size;
  const float image_size_shift =
      image_size + 2 * std::max(std::abs(shift[0]), std::abs(shift[1]));
  std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
      tile_coordinates, nr_tiles, w_step, image_size, image_size_shift,
      padded_tile_size);

  // FFT plans
  typedef std::pair<fftwf_plan, fftwf_plan> fft_pair;
  std::map<size_t, fft_pair> fft_plans;

  // Iterate tiles in batches
  size_t current_nr_tiles = omp_get_max_threads();
  for (size_t tile_offset = 0; tile_offset < static_cast<size_t>(nr_tiles);
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(current_nr_tiles, nr_tiles - tile_offset);

    // Find w_padded_tile_size for current batch
    size_t w_padded_tile_size = *std::max_element(
        w_padded_tile_sizes.begin() + tile_offset,
        w_padded_tile_sizes.begin() + tile_offset + current_nr_tiles);

    // Allocate tile buffers
    xt::xtensor<std::complex<float>, 4> tile_buffers(
        {current_nr_tiles, static_cast<size_t>(nr_polarizations),
         w_padded_tile_size, w_padded_tile_size},
        std::complex<float>(0.0f, 0.0f));

    // Initialize FFT plans
    if (fft_plans.find(w_padded_tile_size) == fft_plans.end()) {
      fftwf_plan plan_forward = fftwf_plan_dft_1d(
          w_padded_tile_size, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
      fftwf_plan plan_backward = fftwf_plan_dft_1d(
          w_padded_tile_size, nullptr, nullptr, FFTW_BACKWARD, FFTW_ESTIMATE);
      fft_plans.insert({w_padded_tile_size, {plan_forward, plan_backward}});
    }

    fft_pair plan_pair = fft_plans.find(w_padded_tile_size)->second;
    fftwf_plan plan_forward = plan_pair.first;
    fftwf_plan plan_backward = plan_pair.second;

    // Split tile from grid
    kernel_tiles_from_grid(
        current_nr_tiles, wtile_size, w_padded_tile_size, nr_polarizations,
        grid_size, &tile_coordinates[tile_offset], tile_buffers.data(), grid);

    // Process the current batch of tiles
#pragma omp parallel for
    for (size_t i = 0; i < current_nr_tiles; i++) {
      unsigned int tile_idx = tile_offset + i;
      const idg::Coordinate& coordinate = tile_coordinates[tile_idx];
      std::complex<float>* tile_ptr = &tile_buffers(i, 0, 0, 0);

      // Backwards FFT
      kernel_fft_composite(plan_backward, nr_polarizations, w_padded_tile_size,
                           tile_ptr);

      // Multiply w term
      kernel_apply_phasor(nr_polarizations, w_padded_tile_size, image_size,
                          w_step, shift, coordinate, tile_ptr, 1);

      // Forward FFT
      kernel_fft_composite(plan_forward, nr_polarizations, w_padded_tile_size,
                           tile_ptr);

      // Copy tile
      size_t dst_idx = index_grid_4d(nr_polarizations, padded_tile_size,
                                     tile_ids[tile_idx], 0, 0, 0);
      kernel_copy_tile(nr_polarizations, w_padded_tile_size, padded_tile_size,
                       tile_ptr, &tiles[dst_idx]);
    }  // end for current_nr_tiles
  }    // end for tile_offset

  // Free FFT plans
  for (auto& entry : fft_plans) {
    fftwf_destroy_plan(entry.second.first);
    fftwf_destroy_plan(entry.second.second);
  }
}  // end kernel_splitter_wtiles_from_grid

}  // end namespace optimized
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg