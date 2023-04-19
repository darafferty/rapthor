// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ThrowAssert.hpp>  // assert
#include <cmath>            // M_PI
#include <climits>
#include <memory>
#include "Proxy.h"

namespace idg {
namespace proxy {
Proxy::Proxy() { m_report.reset(new Report()); }

Proxy::~Proxy() {}

void Proxy::gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array4D<std::complex<float>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterm_offsets,
    const Array2D<float>& spheroidal) {
  assert(m_grid != nullptr);

  check_dimensions(plan.get_options(), plan.get_subgrid_size(), frequencies,
                   visibilities, uvw, baselines, *m_grid, aterms, aterm_offsets,
                   spheroidal);

  if ((plan.get_w_step() != 0.0) &&
      (!do_supports_wstacking() && !do_supports_wtiling())) {
    throw std::invalid_argument(
        "w_step is not zero, but this Proxy does not support gridding with "
        "W-stacking or W-tiling.");
  }

  do_gridding(plan, frequencies, visibilities, uvw, baselines, aterms,
              aterm_offsets, spheroidal);
}

void Proxy::gridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterms_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
  const Array1D<float> frequencies_array(frequencies);
  const Array4D<std::complex<float>> visibilities_array(visibilities);
  const Array2D<UVW<float>> uvw_array(uvw);
  const Array1D<std::pair<unsigned int, unsigned int>> baselines_array(
      baselines);
  const Array4D<Matrix2x2<std::complex<float>>> aterms_array(aterms);
  const Array1D<unsigned int> aterms_offsets_array(aterms_offsets);
  const Array2D<float> taper_array(taper);
  gridding(plan, frequencies_array, visibilities_array, uvw_array,
           baselines_array, aterms_array, aterms_offsets_array, taper_array);
}

void Proxy::degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array4D<std::complex<float>>& visibilities, const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterm_offsets,
    const Array2D<float>& spheroidal) {
  check_dimensions(plan.get_options(), plan.get_subgrid_size(), frequencies,
                   visibilities, uvw, baselines, *m_grid, aterms, aterm_offsets,
                   spheroidal);

  if ((plan.get_w_step() != 0.0) &&
      (!do_supports_wstacking() && !do_supports_wtiling())) {
    throw std::invalid_argument(
        "w_step is not zero, but this Proxy does not support degridding with "
        "W-stacking.");
  }

  do_degridding(plan, frequencies, visibilities, uvw, baselines, aterms,
                aterm_offsets, spheroidal);
}

void Proxy::degridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterms_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
  const Array1D<float> frequencies_array(frequencies);
  Array4D<std::complex<float>> visibilities_array(visibilities);
  const Array2D<UVW<float>> uvw_array(uvw);
  const Array1D<std::pair<unsigned int, unsigned int>> baselines_array(
      baselines);
  const Array4D<Matrix2x2<std::complex<float>>> aterms_array(aterms);
  const Array1D<unsigned int> aterms_offsets_array(aterms_offsets);
  const Array2D<float> taper_array(taper);
  degridding(plan, frequencies_array, visibilities_array, uvw_array,
             baselines_array, aterms_array, aterms_offsets_array, taper_array);
}

void Proxy::calibrate_init(
    const unsigned int kernel_size, const Array2D<float>& frequencies,
    Array4D<std::complex<float>>& visibilities, Array4D<float>& weights,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterm_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // TODO
  // check_dimensions(
  //    subgrid_size, frequencies, visibilities, uvw, baselines,
  //    grid, aterms, aterm_offsets, spheroidal);

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
  auto nr_timesteps = visibilities.get_z_dim();
  auto nr_correlations = visibilities.get_x_dim();
  assert(nr_correlations == 4);
  const size_t nr_baselines = baselines.size();
  auto nr_channel_blocks = frequencies.get_y_dim();
  auto nr_channels_per_block = frequencies.get_x_dim();

  // Initialize
  unsigned int nr_antennas = 0;
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    nr_antennas = max(nr_antennas, baselines(bl).first + 1);
    nr_antennas = max(nr_antennas, baselines(bl).second + 1);
  }

  // New buffers for data grouped by station
  Array3D<UVW<float>> uvw1(nr_antennas, nr_antennas - 1, nr_timesteps);
  Array6D<std::complex<float>> visibilities1(
      nr_antennas, nr_channel_blocks, nr_antennas - 1, nr_timesteps,
      nr_channels_per_block, nr_correlations);
  Array6D<float> weights1(nr_antennas, nr_channel_blocks, nr_antennas - 1,
                          nr_timesteps, nr_channels_per_block, nr_correlations);
  Array2D<std::pair<unsigned int, unsigned int>> baselines1(nr_antennas,
                                                            nr_antennas - 1);

  // Group baselines by station
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    unsigned int antenna1 = baselines(bl).first;
    unsigned int antenna2 = baselines(bl).second;
    if (antenna1 == antenna2) continue;
    unsigned int bl1 = antenna2 - (antenna2 > antenna1);

    baselines1(antenna1, bl1) = {antenna1, antenna2};

    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      for (unsigned int time = 0; time < nr_timesteps; time++) {
        uvw1(antenna1, bl1, time) = uvw(bl, time);
        for (unsigned int channel = 0; channel < nr_channels_per_block;
             channel++) {
          for (unsigned int cor = 0; cor < nr_correlations; cor++) {
            visibilities1(antenna1, channel_block, bl1, time, channel, cor) =
                visibilities(bl, time,
                             channel_block * nr_channels_per_block + channel,
                             cor);
            weights1(antenna1, channel_block, bl1, time, channel, cor) =
                weights(bl, time,
                        channel_block * nr_channels_per_block + channel, cor);
          }
        }
      }
    }

    // Also add swapped baseline
    // Need to conjugate visibilities
    // and invert sign of uvw coordinates

    std::swap(antenna1, antenna2);
    bl1 = antenna2 - (antenna2 > antenna1);
    baselines1(antenna1, bl1) = {antenna1, antenna2};

    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      for (unsigned int time = 0; time < nr_timesteps; time++) {
        uvw1(antenna1, bl1, time).u = -uvw(bl, time).u;
        uvw1(antenna1, bl1, time).v = -uvw(bl, time).v;
        uvw1(antenna1, bl1, time).w = -uvw(bl, time).w;

        for (unsigned int channel = 0; channel < nr_channels_per_block;
             channel++) {
          unsigned int index_cor_transposed[4] = {0, 2, 1, 3};
          for (unsigned int cor = 0; cor < nr_correlations; cor++) {
            visibilities1(antenna1, channel_block, bl1, time, channel, cor) =
                conj(visibilities(
                    bl, time, channel_block * nr_channels_per_block + channel,
                    index_cor_transposed[cor]));
            weights1(antenna1, channel_block, bl1, time, channel, cor) =
                weights(bl, time,
                        channel_block * nr_channels_per_block + channel,
                        index_cor_transposed[cor]);
          }
        }  // end for channel
      }    // end for time
    }
  }  // end for baseline

  // Set Plan options
  Plan::Options options;
  options.nr_w_layers = nr_w_layers;

  // Create one plan per antenna
  std::vector<std::vector<std::unique_ptr<Plan>>> plans(nr_antennas);
  for (unsigned int i = 0; i < nr_antennas; i++) {
    plans[i].reserve(nr_channel_blocks);
    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      const std::array<size_t, 1> frequencies_channel_block_shape{
          nr_channels_per_block};
      auto frequencies_channel_block = aocommon::xt::CreateSpan(
          frequencies.data(channel_block), frequencies_channel_block_shape);
      const std::array<size_t, 1> aterm_offsets_shape{aterm_offsets.size()};
      auto aterm_offsets_span =
          aocommon::xt::CreateSpan(aterm_offsets.data(), aterm_offsets_shape);
      const std::array<size_t, 2> uvw_shape{nr_antennas - 1, nr_timesteps};
      const std::array<size_t, 1> baselines_shape{nr_antennas - 1};
      plans[i].push_back(make_plan(
          kernel_size, frequencies_channel_block,
          aocommon::xt::CreateSpan(uvw1.data(i), uvw_shape),
          aocommon::xt::CreateSpan(baselines1.data(i), baselines_shape),
          aterm_offsets_span, options));
    }
  }

  // Initialize calibration
  do_calibrate_init(std::move(plans), frequencies, std::move(visibilities1),
                    std::move(weights1), std::move(uvw1), std::move(baselines1),
                    spheroidal);
}

void Proxy::calibrate_update(
    const int station_nr, const Array5D<Matrix2x2<std::complex<float>>>& aterms,
    const Array5D<Matrix2x2<std::complex<float>>>& derivative_aterms,
    Array4D<double>& hessian, Array3D<double>& gradient,
    Array1D<double>& residual) {
  do_calibrate_update(station_nr, aterms, derivative_aterms, hessian, gradient,
                      residual);
}

void Proxy::calibrate_finish() { do_calibrate_finish(); }

void Proxy::set_avg_aterm_correction(
    const Array4D<std::complex<float>>& avg_aterm_correction) {
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
    const Array1D<unsigned int>& aterm_offsets, const Array4D<float>& weights,
    idg::Array4D<std::complex<float>>& average_beam) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  do_compute_avg_beam(nr_antennas, nr_channels, uvw, baselines, aterms,
                      aterm_offsets, weights, average_beam);
}

void Proxy::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterm_offsets, const Array4D<float>& weights,
    idg::Array4D<std::complex<float>>& average_beam) {
  average_beam.init(1.0f);
}

void Proxy::check_dimensions(
    const Plan::Options& options, unsigned int subgrid_size,
    unsigned int frequencies_nr_channels,
    unsigned int visibilities_nr_baselines,
    unsigned int visibilities_nr_timesteps,
    unsigned int visibilities_nr_channels,
    unsigned int visibilities_nr_correlations, unsigned int uvw_nr_baselines,
    unsigned int uvw_nr_timesteps, unsigned int uvw_nr_coordinates,
    unsigned int baselines_nr_baselines, unsigned int baselines_two,
    unsigned int grid_nr_polarizations, unsigned int grid_height,
    unsigned int grid_width, unsigned int aterms_nr_timeslots,
    unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
    unsigned int aterms_aterm_width, unsigned int aterms_nr_polarizations,
    unsigned int aterm_offsets_nr_timeslots_plus_one,
    unsigned int spheroidal_height, unsigned int spheroidal_width) const {
  throw_assert(frequencies_nr_channels > 0, "");
  throw_assert(frequencies_nr_channels == visibilities_nr_channels, "");
  throw_assert(visibilities_nr_baselines == uvw_nr_baselines, "");
  throw_assert(visibilities_nr_baselines == baselines_nr_baselines, "");
  throw_assert(visibilities_nr_timesteps == uvw_nr_timesteps, "");
  throw_assert(
      visibilities_nr_correlations == 2 || visibilities_nr_correlations == 4,
      "");
  throw_assert(uvw_nr_coordinates == 3, "");
  throw_assert(baselines_two == 2, "");
  throw_assert(grid_height == grid_width, "");  // TODO: remove restriction
  throw_assert(aterms_nr_timeslots + 1 == aterm_offsets_nr_timeslots_plus_one,
               "");
  throw_assert(aterms_aterm_height == aterms_aterm_width,
               "");  // TODO: remove restriction
  throw_assert(spheroidal_height == subgrid_size, "");
  throw_assert(spheroidal_height == subgrid_size, "");
  if (options.mode == Plan::Mode::FULL_POLARIZATION) {
    throw_assert(visibilities_nr_correlations == 4, "");
    throw_assert(grid_nr_polarizations == 4, "");
    throw_assert(aterms_nr_polarizations == 4, "");
  } else if (options.mode == Plan::Mode::STOKES_I_ONLY) {
    throw_assert(visibilities_nr_correlations == 2, "");
    throw_assert(grid_nr_polarizations == 1, "");
    throw_assert(aterms_nr_polarizations == 4,
                 "");  // TODO: pass only XX and YY
  }
}

void Proxy::check_dimensions(
    const Plan::Options& options, unsigned int subgrid_size,
    const Array1D<float>& frequencies,
    const Array4D<std::complex<float>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterm_offsets,
    const Array2D<float>& spheroidal) const {
  check_dimensions(options, subgrid_size, frequencies.get_x_dim(),
                   visibilities.get_w_dim(), visibilities.get_z_dim(),
                   visibilities.get_y_dim(), visibilities.get_x_dim(),
                   uvw.get_y_dim(), uvw.get_x_dim(), 3, baselines.get_x_dim(),
                   2, grid.get_z_dim(), grid.get_y_dim(), grid.get_x_dim(),
                   aterms.get_w_dim(), aterms.get_z_dim(), aterms.get_y_dim(),
                   aterms.get_x_dim(), 4, aterm_offsets.get_x_dim(),
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
                                           size_t nr_polarizations,
                                           size_t height, size_t width) {
  return std::make_shared<Grid>(nr_w_layers, nr_polarizations, height, width);
}

void Proxy::set_grid(std::shared_ptr<idg::Grid> grid) {
  // Don't create a new shared_ptr when the grid data pointer is
  // the same. This can be the case when the C-interface is used.
  if (!m_grid || m_grid->data() != grid->data()) {
    m_grid = grid;
  }
}

void Proxy::free_grid() { m_grid.reset(); }

std::shared_ptr<Grid> Proxy::get_final_grid() { return m_grid; }

std::unique_ptr<auxiliary::Memory> Proxy::allocate_memory(size_t bytes) {
  return std::unique_ptr<auxiliary::Memory>(
      new auxiliary::DefaultMemory(bytes));
};

}  // end namespace proxy
}  // end namespace idg
