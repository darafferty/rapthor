// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ThrowAssert.hpp>  // assert
#include <cmath>            // M_PI
#include <climits>
#include <memory>
#include "Proxy.h"

namespace idg {
namespace proxy {
Proxy::Proxy()
    : m_avg_aterm_correction(aocommon::xt::CreateSpan<std::complex<float>, 4>(
          nullptr, {0, 0, 0, 0})),
      report_(std::make_shared<Report>()),
      grid_(aocommon::xt::CreateSpan<std::complex<float>, 4>(nullptr,
                                                             {0, 0, 0, 0})) {}

Proxy::~Proxy() {}

void Proxy::gridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
  assert(get_grid().data() != nullptr);

  check_dimensions(plan.get_options(), plan.get_subgrid_size(), frequencies,
                   visibilities, uvw, baselines, get_grid(), aterms,
                   aterm_offsets, taper);

  if ((plan.get_w_step() != 0.0) &&
      (!do_supports_wstacking() && !do_supports_wtiling())) {
    throw std::invalid_argument(
        "w_step is not zero, but this Proxy does not support gridding with "
        "W-stacking or W-tiling.");
  }

  do_gridding(plan, frequencies, visibilities, uvw, baselines, aterms,
              aterm_offsets, taper);
}

void Proxy::degridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
  check_dimensions(plan.get_options(), plan.get_subgrid_size(), frequencies,
                   visibilities, uvw, baselines, get_grid(), aterms,
                   aterm_offsets, taper);

  if ((plan.get_w_step() != 0.0) &&
      (!do_supports_wstacking() && !do_supports_wtiling())) {
    throw std::invalid_argument(
        "w_step is not zero, but this Proxy does not support degridding with "
        "W-stacking.");
  }

  do_degridding(plan, frequencies, visibilities, uvw, baselines, aterms,
                aterm_offsets, taper);
}

void Proxy::calibrate_init(
    const unsigned int kernel_size,
    const aocommon::xt::Span<float, 2>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    aocommon::xt::Span<float, 4>& weights,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // TODO
  // check_dimensions(
  //    subgrid_size, frequencies, visibilities, uvw, baselines,
  //    grid, aterms, aterm_offsets, taper);

  int nr_w_layers;

  if (m_cache_state.w_step != 0.0) {
    if (supports_wtiling()) {
      nr_w_layers = INT_MAX;
    } else if (supports_wstacking()) {
      nr_w_layers = get_grid().shape(0);
    } else {
      throw std::invalid_argument(
          "w_step is not zero, but this Proxy does not support calibration "
          "with W-stacking.");
    }
  } else {
    nr_w_layers = 1;
  }

  // Arguments
  const size_t nr_timesteps = visibilities.shape(1);
  const size_t nr_correlations = visibilities.shape(3);
  assert(nr_correlations == 4);
  const size_t nr_baselines = baselines.size();
  const size_t nr_channel_blocks = frequencies.shape(0);
  const size_t nr_channels_per_block = frequencies.shape(1);

  // Initialize
  unsigned int nr_antennas = 0;
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    nr_antennas = max(nr_antennas, baselines(bl).first + 1);
    nr_antennas = max(nr_antennas, baselines(bl).second + 1);
  }

  // New buffers for data grouped by station
  Tensor<UVW<float>, 3> uvw1 = allocate_tensor<UVW<float>, 3>(
      {nr_antennas, nr_antennas - 1, nr_timesteps});
  Tensor<std::complex<float>, 6> visibilities1 =
      allocate_tensor<std::complex<float>, 6>(
          {nr_antennas, nr_channel_blocks, nr_antennas - 1, nr_timesteps,
           nr_channels_per_block, nr_correlations});
  Tensor<float, 6> weights1 = allocate_tensor<float, 6>(
      {nr_antennas, nr_channel_blocks, nr_antennas - 1, nr_timesteps,
       nr_channels_per_block, nr_correlations});
  Tensor<std::pair<unsigned int, unsigned int>, 2> baselines1 =
      allocate_tensor<std::pair<unsigned int, unsigned int>, 2>(
          {nr_antennas, nr_antennas - 1});

  // Group baselines by station
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    unsigned int antenna1 = baselines(bl).first;
    unsigned int antenna2 = baselines(bl).second;
    if (antenna1 == antenna2) continue;
    unsigned int bl1 = antenna2 - (antenna2 > antenna1);

    baselines1.Span()(antenna1, bl1) = {antenna1, antenna2};

    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      for (unsigned int time = 0; time < nr_timesteps; time++) {
        uvw1.Span()(antenna1, bl1, time) = uvw(bl, time);
        for (unsigned int channel = 0; channel < nr_channels_per_block;
             channel++) {
          for (unsigned int cor = 0; cor < nr_correlations; cor++) {
            visibilities1.Span()(antenna1, channel_block, bl1, time, channel,
                                 cor) =
                visibilities(bl, time,
                             channel_block * nr_channels_per_block + channel,
                             cor);
            weights1.Span()(antenna1, channel_block, bl1, time, channel, cor) =
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
    baselines1.Span()(antenna1, bl1) = {antenna1, antenna2};

    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      for (unsigned int time = 0; time < nr_timesteps; time++) {
        uvw1.Span()(antenna1, bl1, time).u = -uvw(bl, time).u;
        uvw1.Span()(antenna1, bl1, time).v = -uvw(bl, time).v;
        uvw1.Span()(antenna1, bl1, time).w = -uvw(bl, time).w;

        for (unsigned int channel = 0; channel < nr_channels_per_block;
             channel++) {
          unsigned int index_cor_transposed[4] = {0, 2, 1, 3};
          for (unsigned int cor = 0; cor < nr_correlations; cor++) {
            visibilities1.Span()(antenna1, channel_block, bl1, time, channel,
                                 cor) =
                conj(visibilities(
                    bl, time, channel_block * nr_channels_per_block + channel,
                    index_cor_transposed[cor]));
            weights1.Span()(antenna1, channel_block, bl1, time, channel, cor) =
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
          const_cast<float*>(&frequencies(channel_block, 0)),
          frequencies_channel_block_shape);
      const std::array<size_t, 1> aterm_offsets_shape{aterm_offsets.size()};
      auto aterm_offsets_span =
          aocommon::xt::CreateSpan(aterm_offsets.data(), aterm_offsets_shape);
      const std::array<size_t, 2> uvw_shape{nr_antennas - 1, nr_timesteps};
      const std::array<size_t, 1> baselines_shape{nr_antennas - 1};
      plans[i].push_back(make_plan(
          kernel_size, frequencies_channel_block,
          aocommon::xt::CreateSpan(&uvw1.Span()(i, 0, 0), uvw_shape),
          aocommon::xt::CreateSpan(&baselines1.Span()(i, 0), baselines_shape),
          aterm_offsets, options));
    }
  }

  // Initialize calibration
  do_calibrate_init(std::move(plans), frequencies, std::move(visibilities1),
                    std::move(weights1), std::move(uvw1), std::move(baselines1),
                    taper);
}

void Proxy::calibrate_update(
    const int antenna_nr,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
        aterm_derivatives,
    aocommon::xt::Span<double, 4>& hessian,
    aocommon::xt::Span<double, 3>& gradient,
    aocommon::xt::Span<double, 1>& residual) {
  do_calibrate_update(antenna_nr, aterms, aterm_derivatives, hessian, gradient,
                      residual);
}

void Proxy::calibrate_finish() { do_calibrate_finish(); }

void Proxy::set_avg_aterm_correction(
    const aocommon::xt::Span<std::complex<float>, 4>& avg_aterm_correction) {
  m_avg_aterm_correction = avg_aterm_correction;
}

void Proxy::unset_avg_aterm_correction() {
  m_avg_aterm_correction =
      aocommon::xt::CreateSpan<std::complex<float>, 4>(nullptr, {0, 0, 0, 0});
}

void Proxy::transform(DomainAtoDomainB direction) { do_transform(direction); }

void Proxy::transform(DomainAtoDomainB direction, std::complex<float>* grid_ptr,
                      unsigned int grid_nr_correlations,
                      unsigned int grid_height, unsigned int grid_width) {
  throw_assert(grid_height == grid_width, "");  // TODO: remove restriction
  throw_assert(grid_nr_correlations == 1 || grid_nr_correlations == 4, "");

  unsigned int grid_nr_w_layers = 1;  // TODO: make this a parameter

  aocommon::xt::Span<std::complex<float>, 4> grid =
      aocommon::xt::CreateSpan<std::complex<float>, 4>(
          grid_ptr,
          {grid_nr_w_layers, grid_nr_correlations, grid_height, grid_width});

  set_grid(grid);

  do_transform(direction);

  get_final_grid();
}

void Proxy::compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 4>& weights,
    aocommon::xt::Span<std::complex<float>, 4>& average_beam) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  do_compute_avg_beam(nr_antennas, nr_channels, uvw, baselines, aterms,
                      aterm_offsets, weights, average_beam);
}

void Proxy::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 4>& weights,
    aocommon::xt::Span<std::complex<float>, 4>& average_beam) {
  average_beam.fill(std::complex<float>(1.0f, 0.0f));
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
    unsigned int aterm_offsets_nr_timeslots_plus_one, unsigned int taper_height,
    unsigned int taper_width) const {
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
  throw_assert(taper_height == subgrid_size, "");
  throw_assert(taper_height == subgrid_size, "");
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
    const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<std::complex<float>, 4>& grid,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) const {
  check_dimensions(options, subgrid_size, frequencies.size(),
                   visibilities.shape(0), visibilities.shape(1),
                   visibilities.shape(2), visibilities.shape(3), uvw.shape(0),
                   uvw.shape(1), 3, baselines.size(), 2, grid.shape(1),
                   grid.shape(2), grid.shape(3), aterms.shape(0),
                   aterms.shape(1), aterms.shape(2), aterms.shape(3), 4,
                   aterm_offsets.size(), taper.shape(0), taper.shape(1));
}

Tensor<float, 1> Proxy::compute_wavenumbers(
    const aocommon::xt::Span<float, 1>& frequencies) {
  auto wavenumbers = allocate_tensor<float, 1>({frequencies.size()});
  const double speed_of_light = 299792458.0;
  wavenumbers.Span() = 2 * M_PI * frequencies / speed_of_light;
  return wavenumbers;
}

void Proxy::set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) {
  grid_ = grid;
}

void Proxy::free_grid() {
  for (auto memory_iterator = std::begin(memory_);
       memory_iterator != std::end(memory_); ++memory_iterator) {
    if (memory_iterator->get() == reinterpret_cast<void*>(get_grid().data())) {
      memory_.erase(memory_iterator);
      break;
    }
  }
  grid_ =
      aocommon::xt::CreateSpan<std::complex<float>, 4>(nullptr, {0, 0, 0, 0});
}

aocommon::xt::Span<std::complex<float>, 4>& Proxy::get_final_grid() {
  return grid_;
}

std::unique_ptr<auxiliary::Memory> Proxy::allocate_memory(size_t bytes) {
  return std::unique_ptr<auxiliary::Memory>(
      new auxiliary::DefaultMemory(bytes));
};

}  // end namespace proxy
}  // end namespace idg
