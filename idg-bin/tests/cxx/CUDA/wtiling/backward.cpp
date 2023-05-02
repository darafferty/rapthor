// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-cpu.h"
#include "idg-cuda.h"
#include "idg-fft.h"
#include "idg-util.h"

#include "common/Index.h"
#include "common/Math.h"
#include "CUDA/common/CUDA.h"
#include "CUDA/common/InstanceCUDA.h"

#define TOLERANCE 0.0001f

// ids[n]
void init_ids(int* ids, unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    ids[i] = i;
  }
}

// data[][n][m][size][size]
void init_data(std::complex<float>* data, int* ids, unsigned int n,
               unsigned int m, unsigned int size) {
  size_t sizeof_tiles = n * m * size * size * sizeof(std::complex<float>);
  memset((void*)data, 0, sizeof_tiles);
  for (unsigned int tile = 0; tile < n; tile++) {
    float scale = (float)(tile + 1) / n;

    for (unsigned int z = 0; z < m; z++) {
      for (unsigned int y = 0; y < size; y++) {
        for (unsigned int x = 0; x < size; x++) {
          size_t idx = index_grid_4d(m, size, ids[tile], z, y, x);
          data[idx] = std::complex<float>((y + 1), (x + 1)) * scale;
        }
      }
    }
  }
}

double compare_arrays(int n, std::complex<float>* a, std::complex<float>* b) {
  double r_error = 0.0;
  double i_error = 0.0;
  int nnz = 0;

  float r_max = 1;
  float i_max = 1;
  for (int i = 0; i < n; i++) {
    float r_value = abs(a[i].real());
    float i_value = abs(a[i].imag());
    r_max = max(r_value, r_max);
    i_max = max(i_value, i_max);
  }

  for (int i = 0; i < n; i++) {
    std::complex<float> value1 = a[i];
    std::complex<float> value2 = b[i];
    std::complex<float> difference = value1 - value2;

    if (std::abs(value1) > 0 || std::abs(value2)) {
      nnz++;
    }

    if (std::abs(difference.real()) > 0.0f ||
        std::abs(difference.imag() > 0.0f)) {
      if (nnz < 10)
        std::cout << "[" << i << "] " << value1 << " != " << value2
                  << ", difference = " << difference << std::endl;
      r_error += difference.real() * difference.real() / r_max;
      i_error += difference.imag() * difference.imag() / i_max;
    }
  }

  r_error /= max(1, nnz);
  i_error /= max(1, nnz);
  float error = sqrt(r_error + i_error);

  std::cout << "nnz: " << nnz << ", error: " << error << std::endl;

  return error;
}

void subgrids_from_wtiles(const long nr_subgrids, const int nr_polarizations,
                          const int grid_size, const int subgrid_size,
                          const int tile_size, const idg::Metadata* metadata,
                          std::complex<float>* subgrid,
                          const std::complex<float>* tiles) {
#pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();

    for (int s = 0; s < nr_subgrids; s++) {
      // Load subgrid coordinates
      int tile_index = metadata[s].wtile_index;
      int tile_top = metadata[s].wtile_coordinate.x * tile_size -
                     subgrid_size / 2 + grid_size / 2;
      int tile_left = metadata[s].wtile_coordinate.y * tile_size -
                      subgrid_size / 2 + grid_size / 2;

      // position in tile
      int subgrid_x = metadata[s].coordinate.x - tile_top;
      int subgrid_y = metadata[s].coordinate.y - tile_left;

      // Iterate over subgrid rows, starting at a row that belongs to this
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
          float phase = -M_PI * (x + y - subgrid_size) / subgrid_size;
          std::complex<float> phasor(cosf(phase), sinf(phase));

          // Add subgrid value to tiles
          for (int pol = 0; pol < nr_polarizations; pol++) {
            long src_idx =
                index_grid_4d(nr_polarizations, tile_size + subgrid_size,
                              tile_index, pol, y_dst, x_dst);
            long dst_idx = index_subgrid(nr_polarizations, subgrid_size, s, pol,
                                         y_src, x_src);

            subgrid[dst_idx] = phasor * tiles[src_idx];
          }  // end for pol
        }    // end for x
      }      // end for y
    }        // end for s
  }          // end parallel
}  // subgrids_from_wtiles

void wtiles_from_grid(int nr_tiles, int nr_polarizations, int grid_size,
                      int tile_size, int padded_tile_size, int* tile_ids,
                      idg::Coordinate* tile_coordinates,
                      xt::xtensor<std::complex<float>, 4>& tiles,
                      std::complex<float>* grid) {
  // Extract tiles from grid
  for (int i = 0; i < nr_tiles; i++) {
    idg::Coordinate& coordinate = tile_coordinates[i];

    int x0 = coordinate.x * tile_size - (padded_tile_size - tile_size) / 2 +
             grid_size / 2;
    int y0 = coordinate.y * tile_size - (padded_tile_size - tile_size) / 2 +
             grid_size / 2;

    int x_start = std::max(0, x0);
    int y_start = std::max(0, y0);
    int x_end = std::min(x0 + padded_tile_size, grid_size);
    int y_end = std::min(y0 + padded_tile_size, grid_size);

    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        for (int pol = 0; pol < nr_polarizations; pol++) {
          const int index_pol_transposed[4] = {0, 2, 1, 3};
          unsigned int pol_src = index_pol_transposed[pol];
          unsigned long src_idx = index_grid_3d(grid_size, pol_src, y, x);
          tiles(i, pol, y - y0, x - x0) = grid[src_idx];
        }
      }
    }
  }

}  // end wtiles_from_grid

int main(int argc, char* argv[]) {
  // Parameters
  unsigned int nr_stations = 4;
  unsigned int nr_channels = 8;
  unsigned int nr_timesteps = 128;
  unsigned int nr_timeslots = 1;
  unsigned int nr_polarizations = 4;
  unsigned int grid_size = 1024;
  unsigned int subgrid_size = 32;
  unsigned int max_nr_tiles = 16;
  unsigned int tile_size = 128;
  unsigned int padded_tile_size = 160;
  unsigned int kernel_size = 9;
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  float integration_time = 1.0f;
  const char* layout_file = "LOFAR_lba.txt";

  // Initialize data
  idg::Data data = idg::get_example_data(
      nr_baselines, grid_size, integration_time, nr_channels, layout_file);

  // Get remaining parameters
  float image_size = data.compute_image_size(grid_size, nr_channels);
  float cell_size = image_size / grid_size;

  // Initialize data
  const std::array<size_t, 1> frequencies_shape{nr_channels};
  xt::xtensor<float, 1> frequencies(frequencies_shape);
  aocommon::xt::Span<float, 1> frequencies_span =
      aocommon::xt::CreateSpan(frequencies);
  data.get_frequencies(frequencies_span, image_size);
  const std::array<size_t, 2> uvw_shape{nr_baselines, nr_timesteps};
  xt::xtensor<idg::UVW<float>, 2> uvw(uvw_shape);
  auto uvw_span = aocommon::xt::CreateSpan(uvw);
  data.get_uvw(uvw_span);
  xt::xtensor<std::pair<unsigned int, unsigned int>, 1> baselines =
      idg::get_example_baselines(nr_stations, nr_baselines);
  auto baselines_span = aocommon::xt::CreateSpan(baselines);
  xt::xtensor<unsigned int, 1> aterm_offsets({nr_timeslots + 1}, 0);
  auto aterm_offsets_span = aocommon::xt::CreateSpan(aterm_offsets);
  idg::init_example_aterm_offsets(aterm_offsets_span, nr_timesteps);
  std::array<float, 2> shift{10.1f, 20.2f};

  // Set w-terms to zero
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned t = 0; t < nr_timesteps; t++) {
      uvw(bl, t).w = 0.0f;
    }
  }

  // Initialize plan
  idg::WTiles wtiles(max_nr_tiles, tile_size - subgrid_size);
  idg::Plan::Options options;
  options.plan_strict = true;
  idg::Plan plan(kernel_size, subgrid_size, grid_size, cell_size, shift,
                 frequencies_span, uvw_span, baselines_span, aterm_offsets_span,
                 wtiles, options);
  int nr_subgrids = plan.get_nr_subgrids();

  // Get W-Tiling paramters
  auto wtile_info = wtiles.clear();
  auto& tile_ids = wtile_info.wtile_ids;
  auto& tile_coordinates = wtile_info.wtile_coordinates;
  int nr_tiles = tile_coordinates.size();

  // Initialize GPU
  std::cout << ">> Initialize GPU" << std::endl;
  auto info = idg::proxy::cuda::CUDA::default_info();
  cu::init();
  idg::kernel::cuda::InstanceCUDA cuda(info);
  cu::Context& context = cuda.get_context();
  cu::Stream& stream = cuda.get_execute_stream();

  // Size of buffers
  size_t sizeof_tile_ids = nr_tiles * sizeof(int);
  size_t sizeof_tile_coordinates = nr_tiles * sizeof(idg::Coordinate);
  size_t sizeof_tile =
      nr_polarizations * tile_size * tile_size * sizeof(std::complex<float>);
  size_t sizeof_padded_tile = padded_tile_size * padded_tile_size *
                              nr_polarizations * sizeof(std::complex<float>);
  size_t sizeof_subgrid = nr_polarizations * subgrid_size * subgrid_size *
                          sizeof(std::complex<float>);
  size_t sizeof_tiles = max_nr_tiles * sizeof_tile;
  size_t sizeof_padded_tiles = nr_tiles * sizeof_padded_tile;
  size_t sizeof_shift = 2 * sizeof(float);
  size_t sizeof_subgrids = nr_subgrids * sizeof_subgrid;
  size_t sizeof_metadata = plan.get_sizeof_metadata();
  size_t sizeof_grid =
      nr_polarizations * grid_size * grid_size * sizeof(std::complex<float>);

  // Allocate host buffers
  std::cout << ">> Allocate host buffers" << std::endl;
  cu::HostMemory h_padded_tile_ids(context, sizeof_tile_ids);
  cu::HostMemory h_tiles(context, sizeof_tiles);
  cu::HostMemory h_padded_tiles(context, sizeof_padded_tiles);
  cu::HostMemory h_subgrids(context, sizeof_subgrids);

  // Allocate device buffers
  std::cout << ">> Allocate device buffers" << std::endl;
  cu::DeviceMemory d_tile_ids(context, sizeof_tile_ids);
  cu::DeviceMemory d_padded_tile_ids(context, sizeof_tile_ids);
  cu::DeviceMemory d_tile_coordinates(context, sizeof_tile_coordinates);
  cu::DeviceMemory d_tiles(context, sizeof_tiles);
  cu::DeviceMemory d_padded_tiles(context, sizeof_padded_tiles);
  cu::DeviceMemory d_shift(context, sizeof_shift);
  cu::DeviceMemory d_subgrids(context, sizeof_subgrids);
  cu::DeviceMemory d_metadata(context, sizeof_metadata);

  // Allocate unified memory buffers
  std::cout << ">> Allocate unified memory buffers" << std::endl;
  cu::UnifiedMemory u_grid(context, sizeof_grid);

  std::cout << ">> Initialize data" << std::endl;

  // Initialize subgrid ids
  std::vector<int> subgrid_ids(nr_subgrids);
  init_ids(subgrid_ids.data(), nr_subgrids);

  // Initialize tile ids
  init_ids(static_cast<int*>(h_padded_tile_ids.data()), nr_tiles);
  stream.memcpyHtoDAsync(d_tile_ids, tile_ids.data(), sizeof_tile_ids);
  stream.memcpyHtoDAsync(d_padded_tile_ids,
                         static_cast<int*>(h_padded_tile_ids.data()),
                         sizeof_tile_ids);

  // Initalize tile coordinates
  stream.memcpyHtoDAsync(d_tile_coordinates, tile_coordinates.data(),
                         sizeof_tile_coordinates);

  // Initialize tiles
  init_data(static_cast<std::complex<float>*>(h_tiles.data()), tile_ids.data(),
            nr_tiles, nr_polarizations, tile_size);
  stream.memcpyHtoDAsync(d_tiles,
                         static_cast<std::complex<float>*>(h_tiles.data()),
                         h_tiles.size());

  // Init shift
  stream.memcpyHtoDAsync(d_shift, shift.data(), sizeof_shift);

  // Start tests
  std::cout << std::endl;
  double accuracy = 0;
  bool success = true;

  /********************************************************************************
   * Test subgrids to wtiles
   ********************************************************************************/
  std::cout << ">> Testing subgrids_from_wtiles" << std::endl;
  init_data(static_cast<std::complex<float>*>(h_tiles.data()), tile_ids.data(),
            nr_tiles, nr_polarizations, tile_size);
  stream.memcpyHtoDAsync(d_tiles, h_tiles.data(), sizeof_tiles);
  stream.memcpyHtoDAsync(d_metadata, plan.get_metadata_ptr(), sizeof_metadata);

  // Run subgrids_from_wtiles on GPU
  d_subgrids.zero();
  cuda.launch_splitter_subgrids_from_wtiles(
      nr_subgrids, nr_polarizations, grid_size, subgrid_size,
      tile_size - subgrid_size, 0, d_metadata, d_subgrids, d_tiles);
  stream.memcpyDtoHAsync(h_subgrids.data(), d_subgrids, sizeof_subgrids);
  stream.synchronize();

  // Run subgrids_from_wtiles on host
  unsigned int n = nr_subgrids * nr_polarizations * subgrid_size * subgrid_size;

  std::vector<std::complex<float>> subgrids(n);
  stream.synchronize();
  subgrids_from_wtiles(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                       tile_size - subgrid_size, plan.get_metadata_ptr(),
                       subgrids.data(),
                       static_cast<std::complex<float>*>(h_tiles.data()));
  accuracy = compare_arrays(
      n, static_cast<std::complex<float>*>(h_subgrids.data()), subgrids.data());
  if (accuracy < TOLERANCE) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  /********************************************************************************
   * Test wtiles from grid
   ********************************************************************************/
  std::cout << ">> Testing wtiles_from_grid" << std::endl;

  // Initialize grid
  int id = 0;
  init_data(static_cast<std::complex<float>*>(u_grid.data()), &id, 1,
            nr_polarizations, grid_size);

  // Run splitter_wtiles_from_grid on GPU
  d_tiles.zero();
  cuda.launch_splitter_wtiles_from_grid(
      nr_polarizations, nr_tiles, grid_size, tile_size - subgrid_size,
      padded_tile_size, d_padded_tile_ids, d_tile_coordinates, d_padded_tiles,
      u_grid.data());
  stream.memcpyDtoHAsync(h_padded_tiles.data(), d_padded_tiles,
                         sizeof_padded_tiles);
  stream.synchronize();

  // Run splitter_wtiles_from_grid on host
  xt::xtensor<std::complex<float>, 4> padded_tiles(
      {max_nr_tiles, nr_polarizations, padded_tile_size, padded_tile_size},
      std::complex<float>(0.0f, 0.0f));
  wtiles_from_grid(nr_tiles, nr_polarizations, grid_size,
                   tile_size - subgrid_size, padded_tile_size,
                   static_cast<int*>(h_padded_tile_ids.data()),
                   tile_coordinates.data(), padded_tiles,
                   static_cast<std::complex<float>*>(u_grid.data()));

  n = nr_tiles * nr_polarizations * padded_tile_size * padded_tile_size;
  accuracy = compare_arrays(
      n, static_cast<std::complex<float>*>(h_padded_tiles.data()),
      padded_tiles.data());
  if (accuracy < TOLERANCE) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  return !success;
}
