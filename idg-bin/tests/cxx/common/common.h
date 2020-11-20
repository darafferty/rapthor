// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <iomanip>
#include <cstdlib>  // size_t
#include <complex>
#include <limits>

using namespace std;

#include "idg-cpu.h"   // Reference proxy
#include "idg-util.h"  // Data init routines

// computes sqrt(A^2-B^2) / n
float get_accuracy(const int n, const std::complex<float> *A,
                   const std::complex<float> *B) {
  double r_error = 0.0;
  double i_error = 0.0;
  int nnz = 0;

  for (int i = 0; i < n; i++) {
    float r_cmp = A[i].real();
    float i_cmp = A[i].imag();
    float r_ref = B[i].real();
    float i_ref = B[i].imag();
    double r_diff = r_ref - r_cmp;
    double i_diff = i_ref - i_cmp;
    if (abs(B[i]) > 0.0f) {
      nnz++;
      r_error += r_diff * r_diff;
      i_error += i_diff * i_diff;
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
  ostream &os = clog;

  os << "-----------" << endl;
  os << "PARAMETERS:" << endl;

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

  os << "-----------" << endl;
}

// run gridding and degridding for ProxyType and reference CPU
// proxy and compare the outcome; usage run_test<proxy::cpu::Optimized>();
int compare_to_reference(float tol = 1000 *
                                     std::numeric_limits<float>::epsilon()) {
  int info = 0;

  // Parameters
  unsigned int nr_correlations = 4;
  float w_offset = 0;
  unsigned int nr_stations = 9;
  unsigned int nr_channels = 9;
  unsigned int nr_timesteps = 2048;
  unsigned int nr_timeslots = 7;
  unsigned int grid_size = 2048;
  unsigned int subgrid_size = 32;
  unsigned int kernel_size = 9;
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

  // Initialize Data object
  idg::Data data;

  // Determine the max baseline length for given grid_size
  auto max_uv = data.compute_max_uv(grid_size);
  data.print_info();

  // Select only baselines up to max_uv meters long
  data.limit_max_baseline_length(max_uv);
  data.print_info();

  // Restrict the number of baselines to nr_baselines
  data.limit_nr_baselines(nr_baselines);
  data.print_info();

  // Get remaining parameters
  float image_size = data.compute_image_size(grid_size);
  float cell_size = image_size / grid_size;

  // Print parameters
  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size);

  // Initialize proxies
  std::clog << ">>> Initialize proxy" << std::endl;
  ProxyType optimized;
  idg::proxy::cpu::Reference reference;
  std::clog << std::endl;

  // Allocate and initialize data structures
  clog << ">>> Initialize data structures" << endl;
  idg::Array1D<float> frequencies =
      optimized.allocate_array1d<float>(nr_channels);
  data.get_frequencies(frequencies, image_size);
  idg::Array2D<idg::UVW<float>> uvw =
      optimized.allocate_array2d<idg::UVW<float>>(nr_baselines, nr_timesteps);
  data.get_uvw(uvw);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
      idg::get_dummy_visibilities(optimized, nr_baselines, nr_timesteps,
                                  nr_channels);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ref =
      idg::get_dummy_visibilities(reference, nr_baselines, nr_timesteps,
                                  nr_channels);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      idg::get_example_baselines(optimized, nr_stations, nr_baselines);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
      idg::get_example_aterms(optimized, nr_timeslots, nr_stations,
                              subgrid_size, subgrid_size);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(optimized, nr_timeslots, nr_timesteps);
  idg::Array2D<float> spheroidal =
      idg::get_example_spheroidal(optimized, subgrid_size, subgrid_size);
  idg::Array1D<float> shift = idg::get_zero_shift();
  auto grid = optimized.allocate_grid(1, nr_correlations, grid_size, grid_size);
  auto grid_ref =
      reference.allocate_grid(1, nr_correlations, grid_size, grid_size);
  clog << endl;

  // Flag the first visibilities by setting UVW coordinate to infinity
  idg::UVW<float> infinity = {std::numeric_limits<float>::infinity(), 0, 0};
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    for (unsigned int time = 0; time < nr_timesteps / 10; time++) {
      uvw(bl, time) = infinity;
    }
  }

  // Bind the grids to the respective proxies
  optimized.set_grid(grid);
  reference.set_grid(grid_ref);

  // Set w-terms to zero
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned t = 0; t < nr_timesteps; t++) {
      uvw(bl, t).w = 0.0f;
    }
  }

  // Create plan
  clog << ">>> Create plan" << endl;
  idg::Plan::Options options;
  options.plan_strict = true;
  idg::Plan plan(kernel_size, subgrid_size, grid_size, cell_size, frequencies,
                 uvw, baselines, aterms_offsets, options);
  clog << endl;

  // Run gridder
  std::clog << ">>> Run gridding" << std::endl;
  optimized.gridding(plan, w_offset, shift, cell_size, kernel_size,
                     subgrid_size, frequencies, visibilities, uvw, baselines,
                     *grid, aterms, aterms_offsets, spheroidal);
  optimized.get_grid();

  std::clog << ">>> Run reference gridding" << std::endl;
  reference.gridding(plan, w_offset, shift, cell_size, kernel_size,
                     subgrid_size, frequencies, visibilities, uvw, baselines,
                     *grid_ref, aterms, aterms_offsets, spheroidal);
  reference.get_grid();

  float grid_error = get_accuracy(nr_correlations * grid_size * grid_size,
                                  grid->data(), grid_ref->data());

  // Use the same grid for both degridding calls
  reference.set_grid(optimized.get_grid());

  // Run degridder
  std::clog << ">>> Run degridding" << std::endl;
  visibilities.zero();
  visibilities_ref.zero();
  optimized.degridding(plan, w_offset, shift, cell_size, kernel_size,
                       subgrid_size, frequencies, visibilities, uvw, baselines,
                       *grid, aterms, aterms_offsets, spheroidal);

  std::clog << ">>> Run reference degridding" << std::endl;
  reference.degridding(plan, w_offset, shift, cell_size, kernel_size,
                       subgrid_size, frequencies, visibilities_ref, uvw,
                       baselines, *grid, aterms, aterms_offsets, spheroidal);

  std::clog << std::endl;

  // Ignore visibilities that are not included in the plan
  plan.mask_visibilities(visibilities);
  plan.mask_visibilities(visibilities_ref);

  float degrid_error =
      get_accuracy(nr_baselines * nr_timesteps * nr_channels * nr_correlations,
                   (std::complex<float> *)visibilities.data(),
                   (std::complex<float> *)visibilities_ref.data());

  // Report results
  tol = grid_size * grid_size * std::numeric_limits<float>::epsilon();
  if (grid_error < tol) {
    std::cout << "Gridding test PASSED!" << std::endl;
  } else {
    std::cout << "Gridding test FAILED!" << std::endl;
    info = 1;
  }

  tol = nr_baselines * nr_timesteps * nr_channels *
        std::numeric_limits<float>::epsilon();
  if (degrid_error < tol) {
    std::cout << "Degridding test PASSED!" << std::endl;
  } else {
    std::cout << "Degridding test FAILED!" << std::endl;
    info = 2;
  }

  std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
  std::cout << "degrid_error = " << std::scientific << degrid_error
            << std::endl;

  return info;
}
