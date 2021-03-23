// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ThrowAssert.hpp>  // assert
#include <cmath>            // M_PI
#include <climits>
#include <memory>
#include "Proxy.h"

namespace idg {
namespace proxy {
Proxy::Proxy() {}

Proxy::~Proxy() {}

void Proxy::gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
  assert(m_grid != nullptr);

  check_dimensions(plan.get_subgrid_size(), frequencies, visibilities, uvw,
                   baselines, *m_grid, aterms, aterms_offsets, spheroidal);

  if ((plan.get_w_step() != 0.0) &&
      (!do_supports_wstack_gridding() && !do_supports_wtiles())) {
    throw std::invalid_argument(
        "w_step is not zero, but this Proxy does not support gridding with "
        "W-stacking or W-tiling.");
  }

  do_gridding(plan, frequencies, visibilities, uvw, baselines, aterms,
              aterms_offsets, spheroidal);
}

void Proxy::degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
  check_dimensions(plan.get_subgrid_size(), frequencies, visibilities, uvw,
                   baselines, *m_grid, aterms, aterms_offsets, spheroidal);

  if ((plan.get_w_step() != 0.0) &&
      (!do_supports_wstack_degridding() && !do_supports_wtiles())) {
    throw std::invalid_argument(
        "w_step is not zero, but this Proxy does not support degridding with "
        "W-stacking.");
  }

  do_degridding(plan, frequencies, visibilities, uvw, baselines, aterms,
                aterms_offsets, spheroidal);
}

void Proxy::calibrate_init(
    const unsigned int kernel_size, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    Array3D<Visibility<float>>& weights, const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // TODO
  // check_dimensions(
  //    subgrid_size, frequencies, visibilities, uvw, baselines,
  //    grid, aterms, aterms_offsets, spheroidal);

  int nr_w_layers;

  if (m_cache_state.w_step != 0.0) {
    if (supports_wtiling()) {
      nr_w_layers = INT_MAX;
    } else if (supports_wstacking()) {
      nr_w_layers = m_grid->get_w_dim();
    } else {
      throw std::invalid_argument(
          "w_step is not zero, but this Proxy does not support calibration "
          "with W-stacking.");
    }
  } else {
    nr_w_layers = 1;
  }

  // Arguments
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_baselines = baselines.get_x_dim();
  auto nr_channels = frequencies.get_x_dim();

  // Initialize
  unsigned int nr_antennas = 0;
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    nr_antennas = max(nr_antennas, baselines(bl).first + 1);
    nr_antennas = max(nr_antennas, baselines(bl).second + 1);
  }

  // New buffers for data grouped by station
  Array3D<UVW<float>> uvw1(nr_antennas, nr_antennas - 1, nr_timesteps);
  Array4D<Visibility<std::complex<float>>> visibilities1(
      nr_antennas, nr_antennas - 1, nr_timesteps, nr_channels);
  Array4D<Visibility<float>> weights1(nr_antennas, nr_antennas - 1,
                                      nr_timesteps, nr_channels);
  Array2D<std::pair<unsigned int, unsigned int>> baselines1(nr_antennas,
                                                            nr_antennas - 1);

  // Group baselines by station
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    unsigned int antenna1 = baselines(bl).first;
    unsigned int antenna2 = baselines(bl).second;
    if (antenna1 == antenna2) continue;
    unsigned int bl1 = antenna2 - (antenna2 > antenna1);

    baselines1(antenna1, bl1) = {antenna1, antenna2};

    for (unsigned int time = 0; time < nr_timesteps; time++) {
      uvw1(antenna1, bl1, time) = uvw(bl, time);
      for (unsigned int channel = 0; channel < nr_channels; channel++) {
        visibilities1(antenna1, bl1, time, channel) =
            visibilities(bl, time, channel);
        weights1(antenna1, bl1, time, channel) = weights(bl, time, channel);
      }
    }

    // Also add swapped baseline
    // Need to conjugate visibilities
    // and invert sign of uvw coordinates

    std::swap(antenna1, antenna2);
    bl1 = antenna2 - (antenna2 > antenna1);
    baselines1(antenna1, bl1) = {antenna1, antenna2};

    for (unsigned int time = 0; time < nr_timesteps; time++) {
      uvw1(antenna1, bl1, time).u = -uvw(bl, time).u;
      uvw1(antenna1, bl1, time).v = -uvw(bl, time).v;
      uvw1(antenna1, bl1, time).w = -uvw(bl, time).w;

      for (unsigned int channel = 0; channel < nr_channels; channel++) {
        visibilities1(antenna1, bl1, time, channel) = {
            conj(visibilities(bl, time, channel).xx),
            conj(visibilities(bl, time, channel).yx),
            conj(visibilities(bl, time, channel).xy),
            conj(visibilities(bl, time, channel).yy)};
        weights1(antenna1, bl1, time, channel) = {
            weights(bl, time, channel).xx, weights(bl, time, channel).yx,
            weights(bl, time, channel).xy, weights(bl, time, channel).yy};
      }  // end for channel
    }    // end for time
  }      // end for baseline

  // Set Plan options
  Plan::Options options;
  options.nr_w_layers = nr_w_layers;

  // Create one plan per antenna
  std::vector<std::unique_ptr<Plan>> plans;
  plans.reserve(nr_antennas);

  for (unsigned int i = 0; i < nr_antennas; i++) {
    plans.push_back(make_plan(
        kernel_size, frequencies,
        Array2D<UVW<float>>(uvw1.data(i), nr_antennas - 1, nr_timesteps),
        Array1D<std::pair<unsigned int, unsigned int>>(baselines1.data(i),
                                                       nr_antennas - 1),
        aterms_offsets, options));
  }

  // Initialize calibration
  do_calibrate_init(std::move(plans), frequencies, std::move(visibilities1),
                    std::move(weights1), std::move(uvw1), std::move(baselines1),
                    spheroidal);
}

void Proxy::calibrate_update(
    const int station_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
    Array3D<double>& hessian, Array2D<double>& gradient, double& residual) {
  do_calibrate_update(station_nr, aterms, derivative_aterms, hessian, gradient,
                      residual);
}

void Proxy::calibrate_finish() { do_calibrate_finish(); }

void Proxy::calibrate_init_hessian_vector_product() {
  do_calibrate_init_hessian_vector_product();
}

void Proxy::calibrate_update_hessian_vector_product1(
    const int station_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
    const Array2D<float>& parameter_vector) {
  do_calibrate_update_hessian_vector_product1(
      station_nr, aterms, derivative_aterms, parameter_vector);
}

void Proxy::calibrate_update_hessian_vector_product2(
    const int station_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
    Array2D<float>& parameter_vector) {
  do_calibrate_update_hessian_vector_product2(
      station_nr, aterms, derivative_aterms, parameter_vector);
}

void Proxy::set_avg_aterm_correction(
    const Array4D<std::complex<float>>& avg_aterm_correction) {
  if (!supports_avg_aterm_correction()) {
    throw exception::NotImplemented(
        "This proxy does not support average aterm correction");
  }

  // check_dimensions_avg_aterm_correction();
  std::complex<float>* data = avg_aterm_correction.data();
  size_t size =
      avg_aterm_correction.get_x_dim() * avg_aterm_correction.get_y_dim() *
      avg_aterm_correction.get_z_dim() * avg_aterm_correction.get_w_dim();
  m_avg_aterm_correction.resize(size);
  std::copy(data, data + size, m_avg_aterm_correction.begin());
}

void Proxy::unset_avg_aterm_correction() { m_avg_aterm_correction.resize(0); }

void Proxy::transform(DomainAtoDomainB direction) { do_transform(direction); }

void Proxy::transform(DomainAtoDomainB direction, std::complex<float>* grid_ptr,
                      unsigned int grid_nr_correlations,
                      unsigned int grid_height, unsigned int grid_width) {
  throw_assert(grid_height == grid_width, "");  // TODO: remove restriction
  throw_assert(grid_nr_correlations == 1 || grid_nr_correlations == 4, "");

  unsigned int grid_nr_w_layers = 1;  // TODO: make this a parameter

  auto grid =
      std::make_shared<Grid>(grid_ptr, grid_nr_w_layers, grid_nr_correlations,
                             grid_height, grid_width);

  set_grid(grid);

  do_transform(direction);

  get_final_grid();
}

void Proxy::compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets, const Array4D<float>& weights,
    idg::Array4D<std::complex<float>>& average_beam) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  do_compute_avg_beam(nr_antennas, nr_channels, uvw, baselines, aterms,
                      aterms_offsets, weights, average_beam);
}

void Proxy::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets, const Array4D<float>& weights,
    idg::Array4D<std::complex<float>>& average_beam) {
  average_beam.init(1.0f);
}

void Proxy::check_dimensions(
    unsigned int subgrid_size, unsigned int frequencies_nr_channels,
    unsigned int visibilities_nr_baselines,
    unsigned int visibilities_nr_timesteps,
    unsigned int visibilities_nr_channels,
    unsigned int visibilities_nr_correlations, unsigned int uvw_nr_baselines,
    unsigned int uvw_nr_timesteps, unsigned int uvw_nr_coordinates,
    unsigned int baselines_nr_baselines, unsigned int baselines_two,
    unsigned int grid_nr_correlations, unsigned int grid_height,
    unsigned int grid_width, unsigned int aterms_nr_timeslots,
    unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
    unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
    unsigned int aterms_offsets_nr_timeslots_plus_one,
    unsigned int spheroidal_height, unsigned int spheroidal_width) const {
  throw_assert(frequencies_nr_channels > 0, "");
  throw_assert(frequencies_nr_channels == visibilities_nr_channels, "");
  throw_assert(visibilities_nr_baselines == uvw_nr_baselines, "");
  throw_assert(visibilities_nr_baselines == baselines_nr_baselines, "");
  throw_assert(visibilities_nr_timesteps == uvw_nr_timesteps, "");
  throw_assert(
      visibilities_nr_correlations == 1 || visibilities_nr_correlations == 4,
      "");
  throw_assert(visibilities_nr_correlations == grid_nr_correlations, "");
  throw_assert(visibilities_nr_correlations == aterms_nr_correlations, "");
  throw_assert(uvw_nr_coordinates == 3, "");
  throw_assert(baselines_two == 2, "");
  throw_assert(grid_height == grid_width, "");  // TODO: remove restriction
  throw_assert(aterms_nr_timeslots + 1 == aterms_offsets_nr_timeslots_plus_one,
               "");
  throw_assert(aterms_aterm_height == aterms_aterm_width,
               "");  // TODO: remove restriction
  throw_assert(spheroidal_height == subgrid_size, "");
  throw_assert(spheroidal_height == subgrid_size, "");
}

void Proxy::check_dimensions(
    unsigned int subgrid_size, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) const {
  check_dimensions(subgrid_size, frequencies.get_x_dim(),
                   visibilities.get_z_dim(), visibilities.get_y_dim(),
                   visibilities.get_x_dim(), 4, uvw.get_y_dim(),
                   uvw.get_x_dim(), 3, baselines.get_x_dim(), 2,
                   grid.get_z_dim(), grid.get_y_dim(), grid.get_x_dim(),
                   aterms.get_w_dim(), aterms.get_z_dim(), aterms.get_y_dim(),
                   aterms.get_x_dim(), 4, aterms_offsets.get_x_dim(),
                   spheroidal.get_y_dim(), spheroidal.get_x_dim());
}

Array1D<float> Proxy::compute_wavenumbers(
    const Array1D<float>& frequencies) const {
  int nr_channels = frequencies.get_x_dim();
  Array1D<float> wavenumbers(nr_channels);

  const double speed_of_light = 299792458.0;
  for (int i = 0; i < nr_channels; i++) {
    wavenumbers(i) = 2 * M_PI * frequencies(i) / speed_of_light;
  }

  return wavenumbers;
}

std::shared_ptr<Grid> Proxy::allocate_grid(size_t nr_w_layers,
                                           size_t nr_correlations,
                                           size_t height, size_t width) {
  return std::make_shared<Grid>(nr_w_layers, nr_correlations, height, width);
}

void Proxy::set_grid(std::shared_ptr<idg::Grid> grid) {
  // Don't create a new shared_ptr when the grid data pointer is
  // the same. This can be the case when the C-interface is used.
  if (!m_grid || m_grid->data() != grid->data()) {
    m_grid = grid;
  }
}

std::shared_ptr<Grid> Proxy::get_final_grid() { return m_grid; }

std::unique_ptr<auxiliary::Memory> Proxy::allocate_memory(size_t bytes) {
  return std::unique_ptr<auxiliary::Memory>(
      new auxiliary::DefaultMemory(bytes));
};

}  // end namespace proxy
}  // end namespace idg
