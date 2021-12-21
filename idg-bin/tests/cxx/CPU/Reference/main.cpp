// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-cpu.h"   // Reference proxy
#include "idg-util.h"  // Data init routines

#include "common.h"

// Basic idea: write a bunch of test here on the reference code,
// and then make sure that all other implementation conform with
// the reference

using namespace std;

// Compare to analytical solution in case A-terms are identity and w=0
// This test covers the degridder without the A-term and w-terms computation

int test01() {
  int info = 0;

  // Parameters
  unsigned int nr_correlations = 4;
  unsigned int nr_stations = 8;
  unsigned int nr_channels = 9;
  unsigned int nr_timesteps = 2048;
  unsigned int nr_timeslots = 1;
  unsigned int grid_size = 1024;
  unsigned int subgrid_size = 32;
  unsigned int kernel_size = 9;
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  unsigned int nr_w_layers = 1;
  float integration_time = 1.0f;
  const char *layout_file = "LOFAR_lba.txt";

  // Initialize Data object
  idg::Data data = idg::get_example_data(
      nr_baselines, grid_size, integration_time, nr_channels, layout_file);

  // Print data info
  data.print_info();

  // Get remaining parameters
  auto image_size = data.compute_image_size(grid_size, nr_channels);
  double cell_size = image_size / grid_size;

  // Print parameters
  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size);

  // error tolerance, which might need to be adjusted if parameters are changed
  float tol = 0.1f;

  // Allocate and initialize data structures
  clog << ">>> Initialize data structures" << endl;
  idg::Array1D<float> frequencies(nr_channels);
  data.get_frequencies(frequencies, image_size);
  idg::Array2D<idg::UVW<float>> uvw = data.get_uvw(nr_baselines, nr_timesteps);
  idg::Array4D<std::complex<float>> visibilities =
      idg::get_example_visibilities(uvw, frequencies, image_size, grid_size,
                                    nr_correlations);
  idg::Array4D<std::complex<float>> visibilities_ref =
      idg::get_example_visibilities(uvw, frequencies, image_size, grid_size,
                                    nr_correlations);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      idg::get_example_baselines(nr_stations, nr_baselines);
  auto grid = std::make_shared<idg::Grid>(nr_w_layers, nr_correlations,
                                          grid_size, grid_size);
  auto grid_ref = std::make_shared<idg::Grid>(nr_w_layers, nr_correlations,
                                              grid_size, grid_size);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
      idg::get_identity_aterms(nr_timeslots, nr_stations, subgrid_size,
                               subgrid_size);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(nr_timeslots, nr_timesteps);
  idg::Array2D<float> spheroidal =
      idg::get_identity_spheroidal(subgrid_size, subgrid_size);
  idg::Array1D<float> shift = idg::get_zero_shift();
  clog << endl;

  // Set w-terms to zero
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned t = 0; t < nr_timesteps; t++) {
      uvw(bl, t).w = 0.0f;
    }
  }

  // Initialize of center point source
  int offset_x = 80;
  int offset_y = 50;
  int location_x = grid_size / 2 + offset_x;
  int location_y = grid_size / 2 + offset_y;
  float amplitude = 1.0f;
  grid->zero();
  grid_ref->zero();
  (*grid_ref)(0, 0, location_y, location_x) = amplitude;
  (*grid_ref)(0, 1, location_y, location_x) = amplitude;
  (*grid_ref)(0, 2, location_y, location_x) = amplitude;
  (*grid_ref)(0, 3, location_y, location_x) = amplitude;
  visibilities_ref.zero();
  add_pt_src(visibilities_ref, uvw, frequencies, image_size, grid_size,
             offset_x, offset_y, amplitude);
  clog << endl;

  // Initialize proxy
  clog << ">>> Initialize proxy" << endl;
  idg::proxy::cpu::Reference proxy;

  // Set grid
  proxy.set_grid(grid);
  float w_step = 0.0;
  proxy.init_cache(subgrid_size, cell_size, w_step, shift);

  // Create plan
  clog << ">>> Create plan" << endl;
  idg::Plan::Options options;
  options.plan_strict = true;
  std::unique_ptr<idg::Plan> plan = proxy.make_plan(
      kernel_size, frequencies, uvw, baselines, aterms_offsets, options);
  clog << endl;

  // Grid reference visibilities
  clog << ">>> Grid visibilities" << endl;
  proxy.gridding(*plan, frequencies, visibilities_ref, uvw, baselines, aterms,
                 aterms_offsets, spheroidal);
  proxy.transform(idg::FourierDomainToImageDomain);

  float grid_error = get_accuracy(grid_size * grid_size * nr_correlations,
                                  grid->data(), grid_ref->data());

  // Predict visibilities
  clog << ">>> Predict visibilities" << endl;

  proxy.set_grid(grid_ref);
  proxy.transform(idg::ImageDomainToFourierDomain);

  // Set reference grid
  proxy.set_grid(grid_ref);

  proxy.degridding(*plan, frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);
  clog << endl;

  // Compute error
  float degrid_error =
      get_accuracy(nr_baselines * nr_timesteps * nr_channels * nr_correlations,
                   (std::complex<float> *)visibilities.data(),
                   (std::complex<float> *)visibilities_ref.data());

  // Report error
  clog << "Grid error = " << std::scientific << grid_error << endl;
  clog << "Degrid error = " << std::scientific << degrid_error << endl;
  clog << endl;

  // Report gridding results
  if (grid_error < tol) {
    cout << "Gridding test PASSED!" << endl;
  } else {
    cout << "Gridding test FAILED!" << endl;
    info = 1;
  }

  // Report degridding results
  if (degrid_error < tol) {
    cout << "Degridding test PASSED!" << endl;
  } else {
    cout << "Degridding test FAILED!" << endl;
    info = 2;
  }

  return info;
}

int main(int argc, char *argv[]) {
  int info = 0;

  info = test01();
  if (info != 0) return info;

  // info = test02();
  // if (info != 0) return info;

  // info = test03();
  // if (info != 0) return info;

  return info;
}
