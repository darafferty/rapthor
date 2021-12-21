// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * BulkDegridder.cpp
 */

#include "BulkDegridderImpl.h"

#include "BufferSetImpl.h"
#include "common/Plan.h"

#include <algorithm>

namespace idg {
namespace api {

BulkDegridderImpl::BulkDegridderImpl(const BufferSetImpl& bufferset,
                                     const std::vector<double>& frequencies,
                                     const std::size_t nr_stations)
    : bufferset_(bufferset),
      frequencies_(
          bufferset.get_proxy().allocate_array1d<float>(frequencies.size())),
      nr_stations_(nr_stations),
      shift_(bufferset.get_shift().size()) {
  std::copy_n(frequencies.data(), frequencies.size(), frequencies_.data());
  std::copy_n(bufferset.get_shift().data(), shift_.size(), shift_.data());
}

BulkDegridderImpl::~BulkDegridderImpl() {}

void BulkDegridderImpl::compute_visibilities(
    const std::vector<size_t>& antennas1, const std::vector<size_t>& antennas2,
    const std::vector<const double*>& uvws,
    const std::vector<std::complex<float>*>& visibilities,
    const double* uvw_factors, const std::complex<float>* aterms,
    const std::vector<unsigned int>& aterm_offsets) const {
  const std::size_t nr_baselines = antennas1.size();
  const std::size_t nr_timesteps = uvws.size();

  if (antennas1.size() != antennas2.size() ||
      uvws.size() != visibilities.size()) {
    throw std::invalid_argument(
        "DegridderBuffer::compute_visibilities: Invalid vector size.");
  }

  if (aterm_offsets.empty() || aterm_offsets.front() != 0 ||
      aterm_offsets.back() >= nr_timesteps ||
      (!aterms && aterm_offsets.size() > 1)) {
    throw std::invalid_argument(
        "DegridderBuffer::compute_visibilities: Invalid aterms.");
  }

  const size_t subgridsize = bufferset_.get_subgridsize();
  proxy::Proxy& proxy = bufferset_.get_proxy();
  const int nr_correlations = bufferset_.get_nr_correlations();

  static const double kDefaultUVWFactors[3] = {1.0, 1.0, 1.0};
  if (!uvw_factors) uvw_factors = kDefaultUVWFactors;

  auto bufferStationPairs =
      proxy.allocate_array1d<std::pair<unsigned int, unsigned int>>(
          nr_baselines);
  std::vector<int> baseline_map;
  auto bufferUVW =
      proxy.allocate_array2d<UVW<float>>(nr_baselines, nr_timesteps);
  auto bufferVisibilities = proxy.allocate_array4d<std::complex<float>>(
      nr_baselines, nr_timesteps, frequencies_.size(), nr_correlations);
  // The proxy does not touch visibilities for out-of-bound uvw coordinates.
  // Since we copy all visibilities to the caller, initialize them to zero.
  bufferVisibilities.zero();

  bufferStationPairs.init({nr_stations_, nr_stations_});
  baseline_map.reserve(nr_baselines);
  for (size_t bl = 0; bl < nr_baselines; ++bl) {
    const size_t ant1 = antennas1[bl];
    const size_t ant2 = antennas2[bl];
    if (ant1 == ant2) {  // Skip auto-correlations.
      baseline_map.push_back(-1);
    } else {
      const int local_bl = Plan::baseline_index(ant1, ant2, nr_stations_);
      bufferStationPairs(local_bl) = {static_cast<unsigned int>(ant1),
                                      static_cast<unsigned int>(ant2)};
      baseline_map.push_back(local_bl);

      // Transpose input UVW values into bufferUVW, while applying uvw_factors.
      for (size_t t = 0; t < nr_timesteps; ++t) {
        const double* uvw = uvws[t] + bl * 3;
        bufferUVW(local_bl, t) = {static_cast<float>(uvw[0] * uvw_factors[0]),
                                  static_cast<float>(uvw[1] * uvw_factors[1]),
                                  static_cast<float>(uvw[2] * uvw_factors[2])};
      }
    }
  }

  // The proxy expects the number of time steps as last value.
  std::vector<unsigned int> local_aterm_offsets = aterm_offsets;
  local_aterm_offsets.push_back(nr_timesteps);
  const Array1D<unsigned int> aterm_offsets_array(local_aterm_offsets.data(),
                                                  local_aterm_offsets.size());

  // If aterms is empty, create default values and update the pointer.
  std::vector<Matrix2x2<std::complex<float>>> default_aterms;
  if (!aterms) {
    default_aterms.resize(nr_stations_ * subgridsize * subgridsize,
                          {{1}, {0}, {0}, {1}});
    aterms = reinterpret_cast<std::complex<float>*>(default_aterms.data());
  }

  // The const cast is needed since proxy.degridding accepts const references
  // to arrays with non-const values, and the constructor for such arrays only
  // accepts non-const pointers.
  auto* local_aterms = reinterpret_cast<Matrix2x2<std::complex<float>>*>(
      const_cast<std::complex<float>*>(aterms));
  const Array4D<Matrix2x2<std::complex<float>>> aterms_array(
      local_aterms, aterm_offsets_array.get_x_dim() - 1, nr_stations_,
      subgridsize, subgridsize);

  // Set Plan options
  Plan::Options options;
  options.nr_w_layers = proxy.get_grid().get_w_dim();
  options.plan_strict = false;
  options.mode = (bufferset_.get_nr_polarizations() == 4)
                     ? Plan::Mode::FULL_POLARIZATION
                     : Plan::Mode::STOKES_I_ONLY;

  // Create plan
  bufferset_.get_watch(BufferSetImpl::Watch::kPlan).Start();
  std::unique_ptr<Plan> plan =
      proxy.make_plan(bufferset_.get_kernel_size(), frequencies_, bufferUVW,
                      bufferStationPairs, aterm_offsets_array, options);
  bufferset_.get_watch(BufferSetImpl::Watch::kPlan).Pause();

  // Run degridding
  bufferset_.get_watch(BufferSetImpl::Watch::kDegridding).Start();
  proxy.degridding(*plan, frequencies_, bufferVisibilities, bufferUVW,
                   bufferStationPairs, aterms_array, aterm_offsets_array,
                   bufferset_.get_spheroidal());
  bufferset_.get_watch(BufferSetImpl::Watch::kDegridding).Pause();

  // Transpose bufferVisibilities into visibilities.

  const size_t nr_correlations_out = 4;
  const size_t baseline_size = frequencies_.size() * nr_correlations_out;

  if (nr_correlations == nr_correlations_out) {
    for (size_t t = 0; t < nr_timesteps; ++t) {
      for (size_t bl = 0; bl < nr_baselines; ++bl) {
        const int local_bl = baseline_map[bl];
        if (local_bl != -1) {
          const auto* in = reinterpret_cast<std::complex<float>*>(
              bufferVisibilities.data(local_bl, t));
          std::complex<float>* out = visibilities[t] + bl * baseline_size;
          std::copy_n(in, baseline_size, out);
        }
      }
    }
  } else {
    assert(nr_correlations == 2);
    assert(nr_correlations_out == 4);
    const size_t nr_channels = frequencies_.size();

    for (size_t t = 0; t < nr_timesteps; ++t) {
      for (size_t bl = 0; bl < nr_baselines; ++bl) {
        const int local_bl = baseline_map[bl];
        if (local_bl != -1) {
          const auto* in = reinterpret_cast<std::complex<float>*>(
              bufferVisibilities.data(local_bl, t));
          std::complex<float>* out = visibilities[t] + bl * baseline_size;
          for (size_t i = 0; i < nr_channels; ++i) {
            out[i * nr_correlations_out] = in[i * nr_correlations];
            out[i * nr_correlations_out + 1] = 0;
            out[i * nr_correlations_out + 2] = 0;
            out[i * nr_correlations_out + 3] = in[i * nr_correlations + 1];
          }
        }
      }
    }
  }
}

}  // namespace api
}  // namespace idg
