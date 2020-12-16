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

#define PRINT_METADATA 1

std::tuple<int, int, int, int, int, int, int, bool, bool> read_parameters() {
  const unsigned int DEFAULT_NR_STATIONS = 52;
  const unsigned int DEFAULT_NR_CHANNELS = 16;
  const unsigned int DEFAULT_NR_TIMESTEPS = 3600 * 4;
  const unsigned int DEFAULT_NR_TIMESLOTS = DEFAULT_NR_TIMESTEPS / (60 * 30);
  const unsigned int DEFAULT_GRIDSIZE = 4096;
  const unsigned int DEFAULT_SUBGRIDSIZE = 32;
  const bool DEFAULT_USE_WTILES = false;
  const bool DEFAULT_PRINT_METADATA = false;

  char *cstr_nr_stations = getenv("NR_STATIONS");
  auto nr_stations =
      cstr_nr_stations ? atoi(cstr_nr_stations) : DEFAULT_NR_STATIONS;

  char *cstr_nr_channels = getenv("NR_CHANNELS");
  auto nr_channels =
      cstr_nr_channels ? atoi(cstr_nr_channels) : DEFAULT_NR_CHANNELS;

  char *cstr_nr_timesteps = getenv("NR_TIMESTEPS");
  auto nr_timesteps =
      cstr_nr_timesteps ? atoi(cstr_nr_timesteps) : DEFAULT_NR_TIMESTEPS;

  char *cstr_nr_timeslots = getenv("NR_TIMESLOTS");
  auto nr_timeslots =
      cstr_nr_timeslots ? atoi(cstr_nr_timeslots) : DEFAULT_NR_TIMESLOTS;

  char *cstr_grid_size = getenv("GRIDSIZE");
  auto grid_size = cstr_grid_size ? atoi(cstr_grid_size) : DEFAULT_GRIDSIZE;

  char *cstr_subgrid_size = getenv("SUBGRIDSIZE");
  auto subgrid_size =
      cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;

  char *cstr_kernel_size = getenv("KERNELSIZE");
  auto kernel_size =
      cstr_kernel_size ? atoi(cstr_kernel_size) : (subgrid_size / 4) + 1;

  char *cstr_use_wtiles = getenv("USE_WTILES");
  auto use_wtiles =
      cstr_use_wtiles ? atoi(cstr_use_wtiles) : DEFAULT_USE_WTILES;

  char *cstr_print_metadata = getenv("PRINT_METADATA");
  auto print_metadata =
      cstr_print_metadata ? atoi(cstr_print_metadata) : DEFAULT_PRINT_METADATA;

  return std::make_tuple(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                         grid_size, subgrid_size, kernel_size, use_wtiles,
                         print_metadata);
}

void print_parameters(unsigned int nr_stations, unsigned int nr_channels,
                      unsigned int nr_timesteps, unsigned int nr_timeslots,
                      float image_size, unsigned int grid_size,
                      unsigned int subgrid_size, unsigned int kernel_size) {
  const int fw1 = 30;
  const int fw2 = 10;
  std::ostream &os = std::clog;

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

int main(int argc, char **argv) {
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

  // Read parameters from environment
  std::tie(nr_stations, nr_channels, nr_timesteps, nr_timeslots, grid_size,
           subgrid_size, kernel_size, use_wtiles, print_metadata) =
      read_parameters();

  // Compute nr_baselines
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

  // Initialize Data object
  std::clog << ">>> Initialize data" << std::endl;
  idg::Data data = idg::get_example_data(nr_baselines, grid_size, integration_time);

  // Print data info
  data.print_info();

  // Get remaining parameters
  nr_baselines = data.get_nr_baselines();
  float image_size = data.compute_image_size(grid_size);
  float cell_size = image_size / grid_size;

  // Print parameters
  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size);
  std::clog << std::endl;

  // Allocate and initialize data structures
  std::clog << ">>> Initialize data structures" << std::endl;

  // Initialize frequency data
  idg::Array1D<float> frequencies(nr_channels);
  data.get_frequencies(frequencies, image_size);

  // Initalize UVW coordiantes
  idg::Array2D<idg::UVW<float>> uvw(nr_baselines, nr_timesteps);
  data.get_uvw(uvw);

  // Initialize metadata
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      idg::get_example_baselines(nr_stations, nr_baselines);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(nr_timeslots, nr_timesteps);

  // W-Tiles
  idg::WTiles wtiles;
  std::clog << std::endl;

  // Create plan
  std::clog << ">>> Create plan" << std::endl;
  idg::Plan::Options options;
  options.plan_strict = true;
  idg::Plan plan =
      use_wtiles
          ? idg::Plan(kernel_size, subgrid_size, grid_size, cell_size,
                      frequencies, uvw, baselines, aterms_offsets, wtiles,
                      options)
          : idg::Plan(kernel_size, subgrid_size, grid_size, cell_size,
                      frequencies, uvw, baselines, aterms_offsets, options);
  std::clog << std::endl;

  // Report plan
  std::clog << ">>> Plan information" << std::endl;
  auto nr_visibilities_gridded = plan.get_nr_visibilities();
  auto nr_visibilities_total = nr_baselines * nr_timesteps * nr_channels;
  auto percentage_visibility_gridded =
      (float)nr_visibilities_gridded / nr_visibilities_total * 100.0f;
  std::clog << std::fixed << std::setprecision(2);
  std::clog << "Subgrid size:                   " << subgrid_size << std::endl;
  std::clog << "Total number of visibilities:   " << nr_visibilities_total
            << std::endl;
  std::clog << "Gridder number of visibilities: " << nr_visibilities_gridded
            << " (" << percentage_visibility_gridded << " %)" << std::endl;
  std::clog << "Total number of subgrids:       " << plan.get_nr_subgrids()
            << std::endl;

  if (print_metadata) {
    unsigned int nr_subgrids = plan.get_nr_subgrids();
    const idg::Metadata *metadata = plan.get_metadata_ptr();
    for (unsigned i = 0; i < nr_subgrids; i++) {
      idg::Metadata &m = const_cast<idg::Metadata &>(metadata[i]);
      std::cout << m << std::endl;
    }
  }
}
