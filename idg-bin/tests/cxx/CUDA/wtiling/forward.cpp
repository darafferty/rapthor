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
  std::fill_n(data, n * m * size * size, 0);
  for (unsigned int tile = 0; tile < n; tile++) {
    float scale = (float)(tile + 1) / n;

    for (unsigned int z = 0; z < m; z++) {
      for (unsigned int y = 0; y < size; y++) {
        for (unsigned int x = 0; x < size; x++) {
          size_t idx = index_grid_4d(m, size, ids[tile], z, y, x);
          data[idx] =
              std::complex<float>((y + 1) / size, (x + 1) / size) * scale;
        }
      }
    }
  }
}

int compare_tiles(std::complex<float>* tiles, std::complex<float>* padded_tiles,
                  int* tile_ids, int* padded_tile_ids, unsigned int nr_tiles,
                  unsigned int nr_polarizations, unsigned int tile_size,
                  unsigned int padded_tile_size) {
  unsigned int padding = (padded_tile_size - tile_size) / 2;
  unsigned int nr_errors = 0;
  for (unsigned int tile = 0; tile < nr_tiles; tile++) {
    for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
      for (unsigned int y = 0; y < padded_tile_size; y++) {
        for (unsigned int x = 0; x < padded_tile_size; x++) {
          int padded_tile_id = padded_tile_ids[tile];
          int tile_id = tile_ids[tile];

          size_t padded_idx = index_grid_4d(nr_polarizations, padded_tile_size,
                                            padded_tile_id, pol, y, x);
          std::complex<float> value = padded_tiles[padded_idx];

          std::complex<float> reference(0, 0);

          if (y >= padding && y < (tile_size + padding) && x >= padding &&
              x < (tile_size + padding)) {
            size_t idx = index_grid_4d(nr_polarizations, tile_size, tile_id,
                                       pol, y - padding, x - padding);
            reference = tiles[idx];
          }
          std::complex<float> difference = reference - value;
          if (std::abs(difference) != std::complex<float>(0, 0)) {
            if (nr_errors < 10)
              std::cout << "[" << tile << ", " << y << "," << x << "] " << value
                        << " != " << reference << std::endl;
            nr_errors++;
          }
        }
      }
    }
  }

  return nr_errors;
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

void apply_phasor(std::complex<float>* tiles, idg::Coordinate* tile_coordinates,
                  float* shift, unsigned int nr_tiles,
                  unsigned int nr_polarizations, int tile_size,
                  float image_size, float w_step) {
  float cell_size = image_size / tile_size;
  int N = tile_size * tile_size;

  for (unsigned int tile_index = 0; tile_index < nr_tiles; tile_index++) {
    // Compute W
    const idg::Coordinate& coordinate = tile_coordinates[tile_index];
    float w = (coordinate.z + 0.5f) * w_step;

    for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
      for (int y = 0; y < tile_size; y++) {
        for (int x = 0; x < tile_size; x++) {
          // Compute phase
          const float l = (x - (tile_size / 2)) * cell_size;
          const float m = (y - (tile_size / 2)) * cell_size;
          const float n = compute_n(l, -m, shift);
          const float phase = -2 * M_PI * n * w;

          // Compute phasor
          std::complex<float> phasor = {std::cos(phase) / N,
                                        std::sin(phase) / N};
          unsigned int idx =
              index_grid_4d(nr_polarizations, tile_size, tile_index, pol, y, x);

          // Apply phasor
          tiles[idx] *= phasor;
        }
      }
    }
  }
}

void subgrids_to_wtiles(const long nr_subgrids, const int nr_polarizations,
                        const int grid_size, const int subgrid_size,
                        const int tile_size, const idg::Metadata* metadata,
                        const std::complex<float>* subgrid,
                        std::complex<float>* tiles) {
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
          float phase = M_PI * (x + y - subgrid_size) / subgrid_size;
          std::complex<float> phasor(cosf(phase), sinf(phase));

          // Add subgrid value to tiles
          for (int pol = 0; pol < nr_polarizations; pol++) {
            long dst_idx =
                index_grid_4d(nr_polarizations, tile_size + subgrid_size,
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
}  // subgrids_to_wtiles

void wtiles_to_grid(int nr_tiles, int nr_polarizations, int grid_size,
                    int tile_size, int padded_tile_size, int* tile_ids,
                    idg::Coordinate* tile_coordinates,
                    std::complex<float>* tiles,
                    xt::xtensor<std::complex<float>, 3>& grid) {
  for (int i = 0; i < nr_tiles; i++) {
    idg::Coordinate& coordinate = tile_coordinates[i];

    int x0 = coordinate.x * tile_size - (padded_tile_size - tile_size) / 2 +
             grid_size / 2;
    int y0 = coordinate.y * tile_size - (padded_tile_size - tile_size) / 2 +
             grid_size / 2;
    int x_start = std::max(0, x0);
    int y_start = std::max(0, y0);
    int x_end = std::min(x0 + padded_tile_size, (int)grid_size);
    int y_end = std::min(y0 + padded_tile_size, (int)grid_size);

    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        for (int pol = 0; pol < nr_polarizations; pol++) {
          const int index_pol_transposed[nr_polarizations] = {0, 2, 1, 3};
          unsigned int pol_src = pol;
          unsigned int pol_dst = index_pol_transposed[pol];
          unsigned long src_idx =
              index_grid_4d(nr_polarizations, padded_tile_size, i, pol_dst,
                            (y - y0), (x - x0));
          grid(pol_src, y, x) += tiles[src_idx];
        }  // end for pol
      }    // end for x
    }      // end for y
  }        // end for tile_index
}  // end wtiles_to_grid

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
  float w_step = 4.0 / (image_size * image_size);

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
  int nr_errors = 0;
  double accuracy = 0;
  bool success = true;

  /********************************************************************************
   * Test copying tiles forwards and backwards
   ********************************************************************************/
  std::cout << ">> Testing copy_tiles: tiles -> padded tiles" << std::endl;
  cuda.launch_copy_tiles(nr_polarizations, nr_tiles, tile_size,
                         padded_tile_size, d_tile_ids, d_padded_tile_ids,
                         d_tiles, d_padded_tiles);
  stream.memcpyDtoHAsync(h_padded_tiles.data(), d_padded_tiles,
                         sizeof_padded_tiles);
  stream.synchronize();
  nr_errors = compare_tiles(
      static_cast<std::complex<float>*>(h_tiles.data()),
      static_cast<std::complex<float>*>(h_padded_tiles.data()), tile_ids.data(),
      static_cast<int*>(h_padded_tile_ids.data()), nr_tiles, nr_polarizations,
      tile_size, padded_tile_size);
  if (nr_errors == 0) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed with " << nr_errors << " errors." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  std::cout << ">> Testing copy_tiles: padded tiles -> tiles" << std::endl;
  h_tiles.zero();
  d_tiles.zero();
  cuda.launch_copy_tiles(nr_polarizations, nr_tiles, padded_tile_size,
                         tile_size, d_padded_tile_ids, d_tile_ids,
                         d_padded_tiles, d_tiles);
  stream.memcpyDtoHAsync(static_cast<std::complex<float>*>(h_tiles.data()),
                         d_tiles, sizeof_tiles);
  stream.synchronize();
  nr_errors = compare_tiles(
      static_cast<std::complex<float>*>(h_tiles.data()),
      static_cast<std::complex<float>*>(h_padded_tiles.data()), tile_ids.data(),
      static_cast<int*>(h_padded_tile_ids.data()), nr_tiles, nr_polarizations,
      tile_size, padded_tile_size);
  if (nr_errors == 0) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed with " << nr_errors << " errors." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  /********************************************************************************
   * Test applying phase shift to tiles
   ********************************************************************************/
  std::cout << ">> Testing FFT + apply_phasor" << std::endl;

  // Initialize padded tiles
  init_data(static_cast<std::complex<float>*>(h_padded_tiles.data()),
            static_cast<int*>(h_padded_tile_ids.data()), nr_tiles,
            nr_polarizations, padded_tile_size);
  stream.memcpyHtoDAsync(
      d_padded_tiles, static_cast<std::complex<float>*>(h_padded_tiles.data()),
      h_padded_tiles.size());

  // Initialize FFT for padded_tiles
  unsigned stride = 1;
  unsigned dist = padded_tile_size * padded_tile_size;
  unsigned batch = nr_tiles * nr_polarizations;
  cufft::C2C_2D fft(context, padded_tile_size, padded_tile_size, stride, dist,
                    batch);
  fft.setStream(stream);

  // Run apply_phasor on GPU
  cufftComplex* d_padded_tiles_ptr =
      reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_padded_tiles));
  fft.execute(d_padded_tiles_ptr, d_padded_tiles_ptr, CUFFT_INVERSE);
  cuda.launch_apply_phasor_to_wtiles(nr_polarizations, nr_tiles, image_size,
                                     w_step, padded_tile_size, d_padded_tiles,
                                     d_shift, d_tile_coordinates);
  fft.execute(d_padded_tiles_ptr, d_padded_tiles_ptr, CUFFT_FORWARD);
  stream.memcpyDtoHAsync(
      static_cast<std::complex<float>*>(h_padded_tiles.data()), d_padded_tiles,
      sizeof_padded_tiles);
  stream.synchronize();

  // Run apply_phasor on host
  unsigned int n =
      nr_tiles * nr_polarizations * padded_tile_size * padded_tile_size;
  std::vector<std::complex<float>> padded_tiles(n);
  init_data(padded_tiles.data(), static_cast<int*>(h_padded_tile_ids.data()),
            nr_tiles, nr_polarizations, padded_tile_size);
  idg::ifft2f(nr_tiles * nr_polarizations, padded_tile_size, padded_tile_size,
              padded_tiles.data());
  apply_phasor(padded_tiles.data(), tile_coordinates.data(), shift.data(),
               nr_tiles, nr_polarizations, padded_tile_size, image_size,
               w_step);
  idg::fft2f(nr_tiles * nr_polarizations, padded_tile_size, padded_tile_size,
             padded_tiles.data());
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

  /********************************************************************************
   * Test subgrids to wtiles
   ********************************************************************************/
  std::cout << ">> Testing subgrids_to_wtiles" << std::endl;
  init_data(static_cast<std::complex<float>*>(h_subgrids.data()),
            subgrid_ids.data(), nr_subgrids, nr_polarizations, subgrid_size);
  stream.memcpyHtoDAsync(d_subgrids, h_subgrids.data(), sizeof_subgrids);
  stream.memcpyHtoDAsync(d_metadata, plan.get_metadata_ptr(), sizeof_metadata);

  // Run subgrids_to_wtiles on GPU
  d_tiles.zero();
  cuda.launch_adder_subgrids_to_wtiles(nr_subgrids, nr_polarizations, grid_size,
                                       subgrid_size, tile_size - subgrid_size,
                                       0, d_metadata, d_subgrids, d_tiles);
  stream.memcpyDtoHAsync(h_tiles.data(), d_tiles, sizeof_tiles);
  stream.synchronize();

  // Run subgrids_to_wtiles on host
  n = max_nr_tiles * nr_polarizations * tile_size * tile_size;

  std::vector<std::complex<float>> tiles(n);
  stream.synchronize();
  subgrids_to_wtiles(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                     tile_size - subgrid_size, plan.get_metadata_ptr(),
                     static_cast<std::complex<float>*>(h_subgrids.data()),
                     tiles.data());
  accuracy = compare_arrays(
      n, static_cast<std::complex<float>*>(h_tiles.data()), tiles.data());
  if (accuracy < TOLERANCE) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  /********************************************************************************
   * Test wtiles to grid
   ********************************************************************************/
  std::cout << ">> Testing wtiles_to_grid" << std::endl;

  // Initialize tiles
  init_data(static_cast<std::complex<float>*>(h_padded_tiles.data()),
            static_cast<int*>(h_padded_tile_ids.data()), nr_tiles,
            nr_polarizations, padded_tile_size);
  stream.memcpyHtoDAsync(
      d_padded_tiles, static_cast<std::complex<float>*>(h_padded_tiles.data()),
      h_padded_tiles.size());

  // Run adder_wtiles_to_grid on GPU
  cuda.launch_adder_wtiles_to_grid(nr_polarizations, nr_tiles, grid_size,
                                   tile_size - subgrid_size, padded_tile_size,
                                   d_padded_tile_ids, d_tile_coordinates,
                                   d_padded_tiles, u_grid.data());

  // Wait for the computation on the GPU to finish
  stream.synchronize();

  // Run adder_wtiles_to_grid on host
  xt::xtensor<std::complex<float>, 3> grid(
      {nr_polarizations, grid_size, grid_size},
      std::complex<float>(0.0f, 0.0f));
  wtiles_to_grid(
      nr_tiles, nr_polarizations, grid_size, tile_size - subgrid_size,
      padded_tile_size, static_cast<int*>(h_padded_tile_ids.data()),
      tile_coordinates.data(),
      static_cast<std::complex<float>*>(h_padded_tiles.data()), grid);

  n = nr_polarizations * grid_size * grid_size;
  accuracy = compare_arrays(n, static_cast<std::complex<float>*>(u_grid.data()),
                            grid.data());
  if (accuracy < TOLERANCE) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  /********************************************************************************
   * Test wtiles to patch
   ********************************************************************************/
  std::cout << ">> Testing wtiles_to_patch" << std::endl;

  // Initialize tiles
  nr_tiles = 2;
  init_data(static_cast<std::complex<float>*>(h_padded_tiles.data()),
            static_cast<int*>(h_padded_tile_ids.data()), nr_tiles,
            nr_polarizations, padded_tile_size);
  stream.memcpyHtoDAsync(d_padded_tiles, h_padded_tiles.data(),
                         h_padded_tiles.size());

  // Allocate patch
  int patch_size = grid_size / 2;
  size_t sizeof_patch =
      nr_polarizations * patch_size * patch_size * sizeof(std::complex<float>);
  cu::HostMemory h_patch(context, sizeof_patch);
  cu::DeviceMemory d_patch(context, sizeof_patch);

  // Reset grid
  u_grid.zero();

  // Add tiles to patch
  for (unsigned int y = 0; y < grid_size; y += patch_size) {
    for (unsigned int x = 0; x < grid_size; x += patch_size) {
      idg::Coordinate patch_coordinate = {(int)x, (int)y};

      // Reset patch to zero
      d_patch.zero();

      // Run adder_wtiles_to_patch on GPU
      cuda.launch_adder_wtiles_to_patch(
          nr_polarizations, nr_tiles, grid_size, tile_size - subgrid_size,
          padded_tile_size, patch_size, patch_coordinate, d_padded_tile_ids,
          d_tile_coordinates, d_padded_tiles, d_patch);

      // Copy patch to the host
      stream.synchronize();
      stream.memcpyDtoHAsync(h_patch.data(), d_patch, sizeof_patch);

      // Wait for patch to be copied
      stream.synchronize();

      // Add patch to grid
#pragma omp parallel for
      for (int y_ = 0; y_ < patch_size; y_++) {
        for (int x_ = 0; x_ < patch_size; x_++) {
          for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
            std::complex<float>* dst_ptr =
                static_cast<std::complex<float>*>(u_grid.data());
            std::complex<float>* src_ptr =
                static_cast<std::complex<float>*>(h_patch.data());
            size_t dst_idx = index_grid_3d(grid_size, pol, y + y_, x + x_);
            size_t src_idx = index_grid_3d(patch_size, pol, y_, x_);
            dst_ptr[dst_idx] += src_ptr[src_idx];
          }
        }
      }
    }
  }

  // Create reference grid
  grid.fill(std::complex<float>(0.0f, 0.0f));
  wtiles_to_grid(
      nr_tiles, nr_polarizations, grid_size, tile_size - subgrid_size,
      padded_tile_size, static_cast<int*>(h_padded_tile_ids.data()),
      tile_coordinates.data(),
      static_cast<std::complex<float>*>(h_padded_tiles.data()), grid);

  n = nr_polarizations * grid_size * grid_size;
  accuracy = compare_arrays(n, static_cast<std::complex<float>*>(u_grid.data()),
                            grid.data());
  if (accuracy < TOLERANCE) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
    success = false;
  }
  std::cout << std::endl;

  return !success;
}
