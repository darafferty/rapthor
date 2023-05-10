// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <iomanip>
#include <cstdlib>  // size_t
#include <complex>
#include <tuple>
#include <typeinfo>
#include <vector>
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <mutex>

#include "idg-cpu.h"
#include "idg-util.h"  // Data init routines

using namespace std;

std::tuple<int, int, int, int, int, int, int, int, int, int, int, float, bool,
           bool, const char*>
read_parameters() {
  const unsigned int DEFAULT_NR_STATIONS = 52;      // all LOFAR LBA stations
  const unsigned int DEFAULT_NR_CHANNELS = 16 * 4;  // 16 channels, 4 subbands
  const unsigned int DEFAULT_NR_TIMESTEPS =
      (3600 * 4);  // 4 hours of observation
  const unsigned int DEFAULT_NR_TIMESLOTS =
      DEFAULT_NR_TIMESTEPS / (60 * 30);  // update every 30 minutes
  const unsigned int DEFAULT_GRIDSIZE = 4096;
  const unsigned int DEFAULT_SUBGRIDSIZE = 32;
  const unsigned int DEFAULT_NR_CYCLES = 1;
  const float DEFAULT_GRID_PADDING = 1.0;
  const bool DEFAULT_USE_WTILES = false;
  const bool DEFAULT_STOKES_I_ONLY = false;
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

  char* cstr_total_nr_stations = getenv("TOTAL_NR_STATIONS");
  auto total_nr_stations =
      cstr_total_nr_stations ? atoi(cstr_total_nr_stations) : nr_stations;

  char* cstr_total_nr_channels = getenv("TOTAL_NR_CHANNELS");
  auto total_nr_channels =
      cstr_total_nr_channels ? atoi(cstr_total_nr_channels) : nr_channels;

  char* cstr_total_nr_timesteps = getenv("TOTAL_NR_TIMESTEPS");
  auto total_nr_timesteps =
      cstr_total_nr_timesteps ? atoi(cstr_total_nr_timesteps) : nr_timesteps;

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

  char* cstr_nr_cycles = getenv("NR_CYCLES");
  auto nr_cycles = cstr_nr_cycles ? atoi(cstr_nr_cycles) : DEFAULT_NR_CYCLES;

  char* cstr_grid_padding = getenv("GRID_PADDING");
  auto grid_padding =
      cstr_grid_padding ? atof(cstr_grid_padding) : DEFAULT_GRID_PADDING;

  char* cstr_use_wtiles = getenv("USE_WTILES");
  auto use_wtiles =
      cstr_use_wtiles ? atoi(cstr_use_wtiles) : DEFAULT_USE_WTILES;

  char* cstr_stokes_i_only = getenv("STOKES_I_ONLY");
  auto stokes_i_only =
      cstr_stokes_i_only ? atoi(cstr_stokes_i_only) : DEFAULT_STOKES_I_ONLY;

  char* cstr_layout_file = getenv("LAYOUT_FILE");
  const char* layout_file =
      cstr_layout_file ? cstr_layout_file : DEFAULT_LAYOUT_FILE;

  return std::make_tuple(total_nr_stations, total_nr_channels,
                         total_nr_timesteps, nr_stations, nr_channels,
                         nr_timesteps, nr_timeslots, grid_size, subgrid_size,
                         kernel_size, nr_cycles, grid_padding, use_wtiles,
                         stokes_i_only, layout_file);
}

void print_parameters(unsigned int total_nr_stations,
                      unsigned int total_nr_channels,
                      unsigned int total_nr_timesteps, unsigned int nr_stations,
                      unsigned int nr_channels, unsigned int nr_timesteps,
                      unsigned int nr_timeslots, float image_size,
                      unsigned int grid_size, unsigned int subgrid_size,
                      unsigned int kernel_size, float w_step,
                      float grid_padding, bool stokes_i_only) {
  const int fw1 = 30;
  const int fw2 = 10;
  ostream& os = clog;

  os << "-----------" << endl;
  os << "PARAMETERS:" << endl;

  os << setw(fw1) << left << "Total number of stations"
     << "== " << setw(fw2) << right << total_nr_stations << endl;

  os << setw(fw1) << left << "Total number of channels"
     << "== " << setw(fw2) << right << total_nr_channels << endl;

  os << setw(fw1) << left << "Total number of timesteps"
     << "== " << setw(fw2) << right << total_nr_timesteps << endl;

  os << setw(fw1) << left << "Number of stations"
     << "== " << setw(fw2) << right << nr_stations << endl;

  os << setw(fw1) << left << "Number of channels"
     << "== " << setw(fw2) << right << nr_channels << endl;

  os << setw(fw1) << left << "Number of timesteps"
     << "== " << setw(fw2) << right << nr_timesteps << endl;

  os << setw(fw1) << left << "Number of timeslots"
     << "== " << setw(fw2) << right << nr_timeslots << endl;

  os << setw(fw1) << left << "Imagesize"
     << "== " << setw(fw2) << right << image_size << endl;

  os << setw(fw1) << left << "Grid size"
     << "== " << setw(fw2) << right << grid_size << endl;

  os << setw(fw1) << left << "Subgrid size"
     << "== " << setw(fw2) << right << subgrid_size << endl;

  os << setw(fw1) << left << "Kernel size"
     << "== " << setw(fw2) << right << kernel_size << endl;

  os << setw(fw1) << left << "W step size"
     << "== " << setw(fw2) << right << w_step << endl;

  os << setw(fw1) << left << "Grid padding"
     << "== " << setw(fw2) << right << grid_padding << endl;

  os << setw(fw1) << left << "Stokes I only"
     << "== " << setw(fw2) << right << stokes_i_only << endl;

  os << "-----------" << endl;
}

void run() {
  idg::auxiliary::print_version();

  // Constants
  unsigned int nr_w_layers = 1;
  unsigned int nr_correlations = 4;
  unsigned int nr_polarizations = 4;
  unsigned int total_nr_stations;
  unsigned int total_nr_timesteps;
  unsigned int total_nr_channels;
  unsigned int nr_stations;
  unsigned int nr_channels;
  unsigned int nr_timesteps;
  unsigned int nr_timeslots;
  float integration_time = 1.0;
  unsigned int grid_size;
  unsigned int subgrid_size;
  unsigned int kernel_size;
  unsigned int nr_cycles;
  float grid_padding;
  bool use_wtiles;
  bool stokes_i_only;
  const char* layout_file;

  // Read parameters from environment
  std::tie(total_nr_stations, total_nr_channels, total_nr_timesteps,
           nr_stations, nr_channels, nr_timesteps, nr_timeslots, grid_size,
           subgrid_size, kernel_size, nr_cycles, grid_padding, use_wtiles,
           stokes_i_only, layout_file) = read_parameters();
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

  // Update parameters for Stokes-I only mode
  if (stokes_i_only) {
    nr_correlations = 2;
    nr_polarizations = 1;
  }

  // Initialize Data object
  clog << ">>> Initialize data" << endl;
  idg::Data data = idg::get_example_data(
      nr_baselines, grid_size, integration_time, nr_channels, layout_file);

  // Print data info
  data.print_info();

  // Get remaining parameters
  nr_baselines = data.get_nr_baselines();
  float image_size =
      data.compute_image_size(grid_padding * grid_size, nr_channels);
  float cell_size = image_size / grid_size;
  float w_step =
      use_wtiles ? (2 * kernel_size) / (image_size * image_size) : 0.0;

  // Print parameters
  print_parameters(total_nr_stations, total_nr_channels, total_nr_timesteps,
                   nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size, w_step,
                   grid_padding, stokes_i_only);

  // Warn for unrealistic number of timesteps
  float observation_length = (total_nr_timesteps * integration_time) / 3600;
  if (observation_length > 12) {
    clog << "Observation length of: " << observation_length
         << " hours (> 12) selected!" << endl;
  }

  // Initialize proxy
  clog << ">>> Initialize proxy" << endl;
  ProxyType proxy;
  clog << endl;

  // Allocate and initialize static data structures
  clog << ">>> Initialize data structures" << endl;
  aocommon::xt::Span<std::complex<float>, 4> visibilities =
      idg::get_dummy_visibilities(proxy, nr_baselines, nr_timesteps,
                                  nr_channels, nr_correlations);
  aocommon::xt::Span<idg::Matrix2x2<std::complex<float>>, 4> aterms =
      idg::get_identity_aterms(proxy, nr_timeslots, nr_stations, subgrid_size,
                               subgrid_size);
  aocommon::xt::Span<unsigned int, 1> aterm_offsets =
      idg::get_example_aterm_offsets(proxy, nr_timeslots, nr_timesteps);
  aocommon::xt::Span<float, 2> taper =
      idg::get_example_taper(proxy, subgrid_size, subgrid_size);

  aocommon::xt::Span<std::complex<float>, 4> grid =
      proxy.allocate_span<std::complex<float>, 4>(
          {nr_w_layers, nr_polarizations, grid_size, grid_size});
  grid.fill(std::complex<float>(0, 0));
  std::array<float, 2> shift{0.0f, 0.0f};
  aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1> baselines =
      idg::get_example_baselines(proxy, nr_stations, nr_baselines);
  clog << endl;

  // Benchmark
  vector<double> runtimes_gridding;
  vector<double> runtimes_degridding;
  vector<double> runtimes_fft;
  vector<double> runtimes_get_image;
  vector<double> runtimes_imaging;
  size_t total_nr_visibilities = 0;

  // Enable/disable routines
  bool disable_gridding = getenv("DISABLE_GRIDDING");
  bool disable_degridding = getenv("DISABLE_DEGRIDDING");
  bool disable_fft = getenv("DISABLE_FFT");

  // Plan options
  idg::Plan::Options options;
  options.plan_strict = true;
  options.max_nr_timesteps_per_subgrid = 128;
  options.max_nr_channels_per_subgrid = 8;
  options.mode = stokes_i_only ? idg::Plan::Mode::STOKES_I_ONLY
                               : idg::Plan::Mode::FULL_POLARIZATION;
  omp_set_nested(true);

  // Vector of plans
  std::vector<std::unique_ptr<idg::Plan>> plans;

  // Set grid
  proxy.set_grid(grid);

  // Init cache
  proxy.init_cache(subgrid_size, cell_size, w_step, shift);

  // Iterate all cycles
  for (unsigned cycle = 0; cycle < nr_cycles; cycle++) {
    // Start imaging
    double runtime_imaging = -omp_get_wtime();

    // Iterate all time blocks
    unsigned int nr_time_blocks =
        std::ceil((float)total_nr_timesteps / nr_timesteps);
    for (unsigned int t = 0; t < nr_time_blocks; t++) {
      int time_offset = t * nr_timesteps;
      const size_t current_nr_timesteps =
          total_nr_timesteps - time_offset < nr_timesteps
              ? total_nr_timesteps - time_offset
              : nr_timesteps;

      // Initalize UVW coordiantes
      aocommon::xt::Span<idg::UVW<float>, 2> uvw =
          proxy.allocate_span<idg::UVW<float>, 2>(
              {nr_baselines, current_nr_timesteps});

      data.get_uvw(uvw, 0, time_offset, integration_time);

      // Iterate all channel blocks
      for (unsigned channel_offset = 0; channel_offset < total_nr_channels;
           channel_offset += nr_channels) {
        // Report progress
        clog << ">>>" << endl;
        clog << "time: " << time_offset << "-" << time_offset + nr_timesteps
             << ", ";
        clog << "channel: " << channel_offset << "-"
             << channel_offset + nr_channels << endl;
        clog << ">>>" << endl;

        // Initialize frequency data
        aocommon::xt::Span<float, 1> frequencies =
            proxy.allocate_span<float, 1>({nr_channels});
        data.get_frequencies(frequencies, image_size, channel_offset);

        // Create plan
        if (plans.size() == 0 || cycle == 0) {
          plans.emplace_back(proxy.make_plan(kernel_size, frequencies, uvw,
                                             baselines, aterm_offsets,
                                             options));
        }
        idg::Plan& plan = *plans[t];
        total_nr_visibilities += plan.get_nr_visibilities();

        // Run gridding
        clog << ">>> Run gridding" << endl;
        double runtime_gridding = -omp_get_wtime();
        if (!disable_gridding)
          proxy.gridding(plan, frequencies, visibilities, uvw, baselines,
                         aterms, aterm_offsets, taper);
        runtimes_gridding.push_back(runtime_gridding + omp_get_wtime());
        clog << endl;

        // Run degridding
        clog << ">>> Run degridding" << endl;
        double runtime_degridding = -omp_get_wtime();
        if (!disable_degridding)
          proxy.degridding(plan, frequencies, visibilities, uvw, baselines,
                           aterms, aterm_offsets, taper);
        runtimes_degridding.push_back(runtime_degridding + omp_get_wtime());
        clog << endl;
      }  // end for channel_offset
    }    // end for time_offset

    // Only after a call to get_final_grid(), the grid can be used outside of
    // the proxy
    clog << ">>> Get final grid" << endl;
    double runtime_get_image = -omp_get_wtime();
    proxy.get_final_grid();
    runtimes_get_image.push_back(runtime_get_image + omp_get_wtime());
    clog << endl;

    // Run fft
    clog << ">>> Run fft" << endl;
    double runtime_fft = -omp_get_wtime();
    if (!disable_fft) {
      proxy.transform(idg::FourierDomainToImageDomain);
      proxy.transform(idg::ImageDomainToFourierDomain);
    }
    runtimes_fft.push_back(runtime_fft + omp_get_wtime());
    clog << endl;

    // End imaging
    runtimes_imaging.push_back(runtime_imaging + omp_get_wtime());
  }  // end for i (nr_cycles)

  // Compute total runtime
  double runtime_gridding =
      accumulate(runtimes_gridding.begin(), runtimes_gridding.end(), 0.0);
  double runtime_degridding =
      accumulate(runtimes_degridding.begin(), runtimes_degridding.end(), 0.0);
  double runtime_fft =
      accumulate(runtimes_fft.begin(), runtimes_fft.end(), 0.0);
  double runtime_get_image =
      accumulate(runtimes_get_image.begin(), runtimes_get_image.end(), 0.0);
  double runtime_imaging =
      accumulate(runtimes_imaging.begin(), runtimes_imaging.end(), 0.0);

  // Report runtime
  clog << ">>> Total runtime" << endl;
  if (!disable_gridding) idg::report("gridding", runtime_gridding);
  if (!disable_degridding) idg::report("degridding", runtime_degridding);
  if (!disable_fft) idg::report("fft", runtime_fft);
  idg::report("get_image", runtime_get_image);
  idg::report("imaging", runtime_imaging);
  clog << endl;

  // Report throughput
  clog << ">>> Total throughput" << endl;
  if (!disable_gridding)
    idg::report_visibilities("gridding", runtime_gridding,
                             total_nr_visibilities);
  if (!disable_degridding)
    idg::report_visibilities("degridding", runtime_degridding,
                             total_nr_visibilities);
  idg::report_visibilities("imaging", runtime_imaging, total_nr_visibilities);
}
