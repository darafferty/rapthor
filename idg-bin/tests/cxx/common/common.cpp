// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common.h"

#include "idg-util.h"  // Data init routines

#include <iostream>
#include <iomanip>
#include <cstdlib>  // size_t
#include <complex>
#include <limits>

//#define PRINT_ERRORS

// computes sqrt(A^2-B^2) / n
float get_accuracy(const int n, const std::complex<float>* A,
                   const std::complex<float>* B) {
  double r_error = 0.0;
  double i_error = 0.0;
  int nnz = 0;

  float r_max = 1;
  float i_max = 1;
  for (int i = 0; i < n; i++) {
    float r_value = abs(A[i].real());
    float i_value = abs(A[i].imag());
    if (r_value > r_max) {
      r_max = r_value;
    }
    if (i_value > i_max) {
      i_max = i_value;
    }
  }

#if defined(PRINT_ERRORS)
  int nerrors = 0;
#endif

  for (int i = 0; i < n; i++) {
    float r_cmp = A[i].real();
    float i_cmp = A[i].imag();
    float r_ref = B[i].real();
    float i_ref = B[i].imag();
    double r_diff = r_ref - r_cmp;
    double i_diff = i_ref - i_cmp;
    if (abs(B[i]) > 0.0f) {
#if defined(PRINT_ERRORS)
      if ((abs(r_diff) > 0.0f || abs(i_diff) > 0.0f) && nerrors < 16) {
        printf("(%f, %f) - (%f, %f) = (%f, %f)\n", r_cmp, i_cmp, r_ref, i_ref,
               r_diff, i_diff);
        nerrors++;
      }
#endif
      nnz++;
      r_error += (r_diff * r_diff) / r_max;
      i_error += (i_diff * i_diff) / i_max;
    }
  }

#if defined(DEBUG)
  printf("r_error: %f\n", r_error);
  printf("i_error: %f\n", i_error);
  printf("nnz: %d\n", nnz);
#endif

  r_error /= max(1, nnz);
  i_error /= max(1, nnz);

  return sqrt(r_error + i_error);
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

// Run gridding and degridding and compare the outcome.
int compare(idg::proxy::Proxy& proxy1, idg::proxy::Proxy& proxy2, float tol) {
  int info = 0;

  // Parameters
  unsigned int nr_correlations = 4;
  unsigned int nr_polarizations = 4;
  unsigned int nr_stations = 9;
  unsigned int nr_channels = 9;
  unsigned int nr_timesteps = 2048;
  unsigned int nr_timeslots = 7;
  unsigned int grid_size = 2048;
  unsigned int subgrid_size = 32;
  unsigned int kernel_size = 9;
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  float integration_time = 1.0f;
  const char* layout_file = "LOFAR_lba.txt";

  idg::Data data = idg::get_example_data(
      nr_baselines, grid_size, integration_time, nr_channels, layout_file);

  // Get remaining parameters
  float image_size = data.compute_image_size(grid_size, nr_channels);
  float cell_size = image_size / grid_size;

  // Configure Stokes-I only mode
  char* cstr_stokes_i_only = getenv("STOKES_I_ONLY");
  auto stokes_i_only = cstr_stokes_i_only ? atoi(cstr_stokes_i_only) : false;
  if (stokes_i_only) {
    nr_correlations = 2;
    nr_polarizations = 1;
  }

  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size);

  std::clog << ">>> Initialize data structures" << std::endl;
  aocommon::xt::Span<float, 1> frequencies =
      proxy2.allocate_span<float, 1>({nr_channels});
  data.get_frequencies(frequencies, image_size);
  aocommon::xt::Span<idg::UVW<float>, 2> uvw =
      proxy2.allocate_span<idg::UVW<float>, 2>({nr_baselines, nr_timesteps});
  data.get_uvw(uvw);
  aocommon::xt::Span<std::complex<float>, 4> visibilities =
      idg::get_dummy_visibilities(proxy2, nr_baselines, nr_timesteps,
                                  nr_channels, nr_correlations);
  aocommon::xt::Span<std::complex<float>, 4> visibilities_ref =
      idg::get_dummy_visibilities(proxy1, nr_baselines, nr_timesteps,
                                  nr_channels, nr_correlations);
  aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1> baselines =
      idg::get_example_baselines(proxy2, nr_stations, nr_baselines);
  aocommon::xt::Span<idg::Matrix2x2<std::complex<float>>, 4> aterms =
      idg::get_example_aterms(proxy2, nr_timeslots, nr_stations, subgrid_size,
                              subgrid_size);
  aocommon::xt::Span<unsigned int, 1> aterm_offsets =
      idg::get_example_aterm_offsets(proxy2, nr_timeslots, nr_timesteps);
  aocommon::xt::Span<float, 2> taper =
      idg::get_example_taper(proxy2, subgrid_size, subgrid_size);
  // Set shift to some random non-zero value
  // shift is in radians, 0.02rad is about 1deg
  std::array<float, 2> shift{0.02f, -0.03f};
  aocommon::xt::Span<std::complex<float>, 4> grid =
      proxy2.allocate_span<std::complex<float>, 4>(
          {1, nr_polarizations, grid_size, grid_size});
  aocommon::xt::Span<std::complex<float>, 4> grid_ref =
      proxy1.allocate_span<std::complex<float>, 4>(
          {1, nr_polarizations, grid_size, grid_size});
  std::clog << std::endl;

  // Flag the first visibilities by setting UVW coordinate to infinity
  idg::UVW<float> infinity = {std::numeric_limits<float>::infinity(), 0, 0};
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    for (unsigned int time = 0; time < nr_timesteps / 10; time++) {
      uvw(bl, time) = infinity;
    }
  }

  // Bind the grids to the respective proxies
  proxy2.set_grid(grid);
  proxy1.set_grid(grid_ref);

  // Set w-terms to zero
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned t = 0; t < nr_timesteps; t++) {
      uvw(bl, t).w = 0.0f;
    }
  }

  // Init cache, with valid w_steps if both proxies support it
  bool use_wtiles = proxy1.supports_wtiling() && proxy2.supports_wtiling();
  float w_step =
      use_wtiles ? (2 * kernel_size) / (image_size * image_size) : 0.0;
  proxy1.init_cache(subgrid_size, cell_size, w_step, shift);
  proxy2.init_cache(subgrid_size, cell_size, w_step, shift);

  // Create plan
  std::clog << ">>> Create plan" << std::endl;
  idg::Plan::Options options;
  options.plan_strict = true;
  options.mode = stokes_i_only ? idg::Plan::Mode::STOKES_I_ONLY
                               : idg::Plan::Mode::FULL_POLARIZATION;
  std::unique_ptr<idg::Plan> plan1 = proxy1.make_plan(
      kernel_size, frequencies, uvw, baselines, aterm_offsets, options);
  std::unique_ptr<idg::Plan> plan2 = proxy2.make_plan(
      kernel_size, frequencies, uvw, baselines, aterm_offsets, options);
  std::clog << std::endl;

#if TEST_GRIDDING
  // Run gridder
  std::clog << ">>> Run gridding" << std::endl;
  grid.fill(std::complex<float>(0, 0));
  proxy2.gridding(*plan2, frequencies, visibilities, uvw, baselines, aterms,
                  aterm_offsets, taper);
  proxy2.get_final_grid();

  std::clog << ">>> Run reference gridding" << std::endl;
  grid_ref.fill(std::complex<float>(0.0f, 0.0f));
  proxy1.gridding(*plan1, frequencies, visibilities, uvw, baselines, aterms,
                  aterm_offsets, taper);
  proxy1.get_final_grid();

  float grid_error = get_accuracy(nr_polarizations * grid_size * grid_size,
                                  grid.data(), grid_ref.data());
#endif

  // Use the same grid for both degridding calls
  proxy2.set_grid(grid);
  proxy1.set_grid(grid);

#if TEST_DEGRIDDING
  // Run degridder
  std::clog << ">>> Run degridding" << std::endl;
  visibilities.fill(std::complex<float>(0.0f, 0.0f));
  proxy2.degridding(*plan2, frequencies, visibilities, uvw, baselines, aterms,
                    aterm_offsets, taper);

  std::clog << ">>> Run reference degridding" << std::endl;
  visibilities_ref.fill(std::complex<float>(0.0f, 0.0f));
  proxy1.degridding(*plan1, frequencies, visibilities_ref, uvw, baselines,
                    aterms, aterm_offsets, taper);

  float degrid_error =
      get_accuracy(nr_baselines * nr_timesteps * nr_channels * nr_correlations,
                   (std::complex<float>*)visibilities.data(),
                   (std::complex<float>*)visibilities_ref.data());
#endif

#if TEST_AVERAGE_BEAM
  auto average_beam = proxy2.allocate_span<std::complex<float>, 4>(
      {subgrid_size, subgrid_size, 4, 4});
  auto average_beam_ref = proxy1.allocate_span<std::complex<float>, 4>(
      {subgrid_size, subgrid_size, 4, 4});
  auto weights = proxy1.allocate_span<float, 4>(
      {nr_baselines, nr_timesteps, nr_channels, 4});
  weights.fill(1);
  average_beam.fill(std::complex<float>(0, 0));
  average_beam_ref.fill(std::complex<float>(0, 0));
  proxy1.compute_avg_beam(nr_stations, nr_channels, uvw, baselines, aterms,
                          aterm_offsets, weights, average_beam_ref);
  proxy2.compute_avg_beam(nr_stations, nr_channels, uvw, baselines, aterms,
                          aterm_offsets, weights, average_beam);
  float average_beam_error =
      get_accuracy(subgrid_size * subgrid_size * 4 * 4,
                   (std::complex<float>*)average_beam.data(),
                   (std::complex<float>*)average_beam_ref.data());
#endif

  // Report results
#if TEST_GRIDDING
  tol = grid_size * grid_size * std::numeric_limits<float>::epsilon();
  if (grid_error < tol) {
    std::cout << "Gridding test PASSED!" << std::endl;
  } else {
    std::cout << "Gridding test FAILED!" << std::endl;
    info = 1;
  }
#endif

#if TEST_DEGRIDDING
  tol = nr_baselines * nr_timesteps * nr_channels *
        std::numeric_limits<float>::epsilon();
  if (degrid_error < tol) {
    std::cout << "Degridding test PASSED!" << std::endl;
  } else {
    std::cout << "Degridding test FAILED!" << std::endl;
    info = 2;
  }
#endif

#if TEST_AVERAGE_BEAM
  tol = subgrid_size * subgrid_size * 4 * 4 *
        std::numeric_limits<float>::epsilon();
  if (average_beam_error < tol) {
    std::cout << "Average beam test PASSED!" << std::endl;
  } else {
    std::cout << "Average beam test FAILED!" << std::endl;
    info = 3;
  }
#endif

#if TEST_GRIDDING
  std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
#endif
#if TEST_DEGRIDDING
  std::cout << "degrid_error = " << std::scientific << degrid_error
            << std::endl;
#endif
#if TEST_AVERAGE_BEAM
  std::cout << "average_beam_error = " << std::scientific << average_beam_error
            << std::endl;
#endif

  return info;
}
