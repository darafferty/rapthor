// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <iomanip>
#include <cstdlib>  // size_t
#include <complex>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "idg-cpu.h"
#include "idg-util.h"  // Data init routines

std::tuple<int, int, int, int, int, int, int, bool, bool, const char*>
read_parameters() {
  const unsigned int DEFAULT_NR_STATIONS = 52;
  const unsigned int DEFAULT_NR_CHANNELS = 16;
  const unsigned int DEFAULT_NR_TIMESTEPS = 3600 * 4;
  const unsigned int DEFAULT_NR_TIMESLOTS = DEFAULT_NR_TIMESTEPS / (60 * 30);
  const unsigned int DEFAULT_GRIDSIZE = 4096;
  const unsigned int DEFAULT_SUBGRIDSIZE = 32;
  const bool DEFAULT_USE_WTILES = false;
  const bool DEFAULT_PRINT_METADATA = false;
  const char* DEFAULT_LAYOUT_FILE = "LOFAR_lba.txt";

  char* cstr_nr_stations = getenv("NR_STATIONS");
  auto nr_stations =
      cstr_nr_stations ? atoi(cstr_nr_stations) : DEFAULT_NR_STATIONS;

  char* cstr_nr_channels = getenv("NR_CHANNELS");
  auto nr_channels =
      cstr_nr_channels ? atoi(cstr_nr_channels) : DEFAULT_NR_CHANNELS;

  char* cstr_nr_timesteps = getenv("NR_TIMESTEPS");
  auto nr_timesteps =
      cstr_nr_timesteps ? atoi(cstr_nr_timesteps) : DEFAULT_NR_TIMESTEPS;

  char* cstr_nr_timeslots = getenv("NR_TIMESLOTS");
  auto nr_timeslots =
      cstr_nr_timeslots ? atoi(cstr_nr_timeslots) : DEFAULT_NR_TIMESLOTS;

  char* cstr_grid_size = getenv("GRIDSIZE");
  auto grid_size = cstr_grid_size ? atoi(cstr_grid_size) : DEFAULT_GRIDSIZE;

  char* cstr_subgrid_size = getenv("SUBGRIDSIZE");
  auto subgrid_size =
      cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;

  char* cstr_kernel_size = getenv("KERNELSIZE");
  auto kernel_size =
      cstr_kernel_size ? atoi(cstr_kernel_size) : (subgrid_size / 4) + 1;

  char* cstr_use_wtiles = getenv("USE_WTILES");
  auto use_wtiles =
      cstr_use_wtiles ? atoi(cstr_use_wtiles) : DEFAULT_USE_WTILES;

  char* cstr_print_metadata = getenv("PRINT_METADATA");
  auto print_metadata =
      cstr_print_metadata ? atoi(cstr_print_metadata) : DEFAULT_PRINT_METADATA;

  char* cstr_layout_file = getenv("LAYOUT_FILE");
  const char* layout_file =
      cstr_layout_file ? cstr_layout_file : DEFAULT_LAYOUT_FILE;

  return std::make_tuple(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                         grid_size, subgrid_size, kernel_size, use_wtiles,
                         print_metadata, layout_file);
}

void print_parameters(unsigned int nr_stations, unsigned int nr_channels,
                      unsigned int nr_timesteps, unsigned int nr_timeslots,
                      float image_size, unsigned int grid_size,
                      unsigned int subgrid_size, unsigned int kernel_size) {
  const int fw1 = 30;
  const int fw2 = 10;
  std::ostream& os = std::clog;

  os << "-----------" << std::endl;
  os << "PARAMETERS:" << std::endl;

  os << std::setw(fw1) << std::left << "Number of stations"
     << "== " << std::setw(fw2) << std::right << nr_stations << std::endl;

  os << std::setw(fw1) << std::left << "Number of channels"
     << "== " << std::setw(fw2) << std::right << nr_channels << std::endl;

  os << std::setw(fw1) << std::left << "Number of timesteps"
     << "== " << std::setw(fw2) << std::right << nr_timesteps << std::endl;

  os << std::setw(fw1) << std::left << "Number of timeslots"
     << "== " << std::setw(fw2) << std::right << nr_timeslots << std::endl;

  os << std::setw(fw1) << std::left << "Imagesize"
     << "== " << std::setw(fw2) << std::right << image_size << std::endl;

  os << std::setw(fw1) << std::left << "Grid size"
     << "== " << std::setw(fw2) << std::right << grid_size << std::endl;

  os << std::setw(fw1) << std::left << "Subgrid size"
     << "== " << std::setw(fw2) << std::right << subgrid_size << std::endl;

  os << std::setw(fw1) << std::left << "Kernel size"
     << "== " << std::setw(fw2) << std::right << kernel_size << std::endl;

  os << "-----------" << std::endl;
}

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

int main(int argc, char** argv) {
  // Constants
  unsigned int nr_stations;
  unsigned int nr_channels;
  unsigned int nr_timesteps;
  unsigned int nr_timeslots;
  unsigned int grid_size;
  unsigned int subgrid_size;
  unsigned int kernel_size;
  bool use_wtiles;
  bool print_metadata;
  float integration_time = 1.0f;
  unsigned int nr_polarizations = 4;
  std::string layout_file;

  // Read parameters from environment
  std::tie(nr_stations, nr_channels, nr_timesteps, nr_timeslots, grid_size,
           subgrid_size, kernel_size, use_wtiles, print_metadata, layout_file) =
      read_parameters();

  // Compute nr_baselines
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

  // Initialize Data object
  std::clog << ">>> Initialize data" << std::endl;
  idg::Data data = idg::get_example_data(
      nr_baselines, grid_size, integration_time, nr_channels, layout_file);

  // Print data info
  data.print_info();

  // Get remaining parameters
  nr_baselines = data.get_nr_baselines();
  float image_size = data.compute_image_size(grid_size, nr_channels);
  float cell_size = image_size / grid_size;

  // Print parameters
  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size);
  std::clog << std::endl;

  // Allocate and initialize data structures
  std::clog << ">>> Initialize data structures" << std::endl;

  // Initialize shift
  std::array<float, 2> shift{0.0f, 0.0f};

  // Initialize proxy
  unsigned int nr_correlations = 4;
  idg::proxy::cpu::Optimized proxy;
  aocommon::xt::Span<std::complex<float>, 4> grid =
      proxy.allocate_span<std::complex<float>, 4>(
          {1, nr_correlations, grid_size, grid_size});
  proxy.set_grid(grid);
  float w_step = use_wtiles ? 4.0 / (image_size * image_size) : 0.0;

  proxy.init_cache(subgrid_size, cell_size, w_step, shift);

  // Initalize UVW coordinates
  aocommon::xt::Span<idg::UVW<float>, 2> uvw =
      proxy.allocate_span<idg::UVW<float>, 2>({nr_baselines, nr_timesteps});
  data.get_uvw(uvw);

  // Initialize frequency data
  auto frequencies = proxy.allocate_span<float, 1>({nr_channels});
  data.get_frequencies(frequencies, image_size);

  // Initialize metadata
  aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1> baselines =
      idg::get_example_baselines(proxy, nr_stations, nr_baselines);
  aocommon::xt::Span<unsigned int, 1> aterm_offsets =
      idg::get_example_aterm_offsets(proxy, nr_timeslots, nr_timesteps);

  // Create plan
  std::clog << ">>> Create plan" << std::endl;
  idg::Plan::Options options;
  options.plan_strict = true;
  auto plan = proxy.make_plan(kernel_size, frequencies, uvw, baselines,
                              aterm_offsets, options);
  std::clog << std::endl;

  // Report plan
  std::clog << ">>> Plan information" << std::endl;
  auto nr_visibilities_gridded = plan->get_nr_visibilities();
  auto nr_visibilities_total = nr_baselines * nr_timesteps * nr_channels;
  auto percentage_visibility_gridded =
      (float)nr_visibilities_gridded / nr_visibilities_total * 100.0f;
  std::clog << std::fixed << std::setprecision(2);
  std::clog << "Subgrid size:                   " << subgrid_size << std::endl;
  std::clog << "Total number of visibilities:   " << nr_visibilities_total
            << std::endl;
  std::clog << "Gridder number of visibilities: " << nr_visibilities_gridded
            << " (" << percentage_visibility_gridded << " %)" << std::endl;
  std::clog << "Total number of subgrids:       " << plan->get_nr_subgrids()
            << std::endl;

  if (print_metadata) {
    unsigned int nr_subgrids = plan->get_nr_subgrids();
    const idg::Metadata* metadata = plan->get_metadata_ptr();
    for (unsigned i = 0; i < nr_subgrids; i++) {
      std::cout << metadata[i] << std::endl;
    }
  }

  // W-Tile info
  if (use_wtiles) {
    std::clog << std::endl;
    std::clog << ">>> W-Tile information" << std::endl;

    // Get a map with all the tile coordinates
    idg::WTileMap tile_map;
    unsigned int nr_subgrids = plan->get_nr_subgrids();
    const idg::Metadata* metadata = plan->get_metadata_ptr();
    for (unsigned i = 0; i < nr_subgrids; i++) {
      idg::Metadata& m = const_cast<idg::Metadata&>(metadata[i]);
      idg::WTileInfo tile_info;
      tile_map[m.wtile_coordinate] = tile_info;
    }

    // Convert WTileMap to vector of tile coordinates
    std::vector<idg::Coordinate> tile_coordinates;
    for (const auto tile_info : tile_map) {
      tile_coordinates.push_back(tile_info.first);
    }

    // W-Tiling parameters
    int nr_tiles = tile_coordinates.size();
    int wtile_size = 128;
    float max_abs_w = 0.0;
    for (auto& tile_coordinate : tile_coordinates) {
      float w = (tile_coordinate.z + 0.5f) * w_step;
      max_abs_w = std::max(max_abs_w, std::abs(w));
    }
    int padded_tile_size = wtile_size + subgrid_size;
    int w_padded_tile_size = next_composite(
        padded_tile_size + int(ceil(max_abs_w * image_size * image_size)));

    // Compute the size of some data structures
    size_t sizeof_padded_tile = 1ULL * padded_tile_size * padded_tile_size *
                                nr_polarizations * sizeof(std::complex<float>);
    size_t sizeof_padded_tiles = 1ULL * nr_tiles * sizeof_padded_tile;
    float sizeof_padded_tiles_gb =
        (float)sizeof_padded_tiles / (1024 * 1024 * 1024);
    size_t sizeof_grid = 1ULL * grid_size * grid_size * nr_polarizations *
                         sizeof(std::complex<float>);
    float sizeof_grid_gb = (float)sizeof_grid / (1024 * 1024 * 1024);

    // Print information
    float nr_subgrids_per_tile = nr_subgrids / nr_tiles;
    std::clog << "tile_size            : " << wtile_size << std::endl;
    std::clog << "nr_tiles             : " << nr_tiles << std::endl;
    std::clog << "nr_subgrids_per_tile : " << nr_subgrids_per_tile << std::endl;
    std::clog << "padded_tile_size     : " << padded_tile_size << std::endl;
    std::clog << "w_padded_tile_size   : " << w_padded_tile_size << std::endl;
    std::clog << "sizeof_grid          : " << sizeof_grid_gb << " Gb"
              << std::endl;
    std::clog << "sizeof_padded_tiles  : " << sizeof_padded_tiles_gb << " Gb"
              << std::endl;

    // Count the number of grid pixels updated,
    // and the total number of pixel updates.
    size_t nr_pixels_updated = 0;
    size_t nr_pixel_updates = 0;

#pragma omp parallel for reduction(+:nr_pixels_updated) reduction(+:nr_pixel_updates)
    for (int y = 0; y < (int)grid_size; y++) {
      std::vector<int> row(grid_size);

      for (const auto& tile_coordinate : tile_coordinates) {
        int x0 = tile_coordinate.x * wtile_size -
                 (padded_tile_size - wtile_size) / 2 + grid_size / 2;
        int y0 = tile_coordinate.y * wtile_size -
                 (padded_tile_size - wtile_size) / 2 + grid_size / 2;
        int x_start = std::max(0, x0);
        int y_start = std::max(0, y0);
        int x_end = std::min(x0 + padded_tile_size, (int)grid_size);
        int y_end = std::min(y0 + padded_tile_size, (int)grid_size);

        if (y >= y_start && y < y_end) {
          for (int x = x_start; x < x_end; x++) {
            row[x]++;
          }
        }
      }

      // Count the number of pixels updates
      for (int x = 0; x < (int)grid_size; x++) {
        nr_pixels_updated += row[x] > 0;
        nr_pixel_updates += row[x];
      }
    }

    // Compute and print statistics
    std::clog << std::fixed << std::setprecision(2);

    // The number of tile pixels per grid pixel
    float sizeof_ratio = (float)sizeof_padded_tiles / sizeof_grid;
    std::clog << "sizeof_ratio         : " << sizeof_ratio * 100 << " %"
              << std::endl;

    // Fraction of all grid pixels updated
    float update_ratio = ((double)nr_pixels_updated / grid_size) / grid_size;
    std::clog << "update_ratio         : " << update_ratio * 100 << " %"
              << std::endl;

    // The average number of times that an updated pixel is touched
    float nr_updates_per_pixel = (float)nr_pixel_updates / nr_pixels_updated;
    std::clog << "nr_updates_per_pixel : " << nr_updates_per_pixel << std::endl;
  }
}
