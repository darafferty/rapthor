// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * GridderBuffer.h
 * Access to IDG's high level gridder routines
 */

#include "GridderBufferImpl.h"
#include "BufferSetImpl.h"

#include <mutex>
#include <csignal>

#include <omp.h>

namespace idg {
namespace api {

GridderBufferImpl::GridderBufferImpl(const BufferSetImpl& bufferset,
                                     size_t bufferTimesteps)
    : BufferImpl(bufferset, bufferTimesteps),
      m_bufferUVW2(
          aocommon::xt::CreateSpan<idg::UVW<float>, 2>(nullptr, {0, 0})),
      m_bufferStationPairs2(
          aocommon::xt::CreateSpan<std::pair<unsigned int, unsigned int>, 1>(
              nullptr, {0})),
      m_bufferVisibilities2(aocommon::xt::CreateSpan<std::complex<float>, 4>(
          nullptr, {0, 0, 0, 0})),
      m_buffer_weights(
          aocommon::xt::CreateSpan<float, 4>(nullptr, {0, 0, 0, 0})),
      m_buffer_weights2(
          aocommon::xt::CreateSpan<float, 4>(nullptr, {0, 0, 0, 0})),
      m_average_beam(nullptr) {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif
  m_aterm_offsets2 = m_default_aterm_offsets;
}

GridderBufferImpl::~GridderBufferImpl() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif
  if (m_flush_thread.joinable()) m_flush_thread.join();
}

void GridderBufferImpl::grid_visibilities(
    size_t timeIndex, size_t antenna1, size_t antenna2,
    const double* uvwInMeters, const std::complex<float>* visibilities,
    const float* weights) {
  // exclude auto-correlations
  if (antenna1 == antenna2) return;

  int local_time = timeIndex - m_timeStartThisBatch;
  size_t local_bl = Plan::baseline_index(antenna1, antenna2, m_nrStations);

  if (local_time < 0) {
    m_timeStartThisBatch = 0;
    m_timeStartNextBatch = m_bufferTimesteps;
    local_time = timeIndex;
  }

  if (local_time >= m_bufferTimesteps) {
    // Empty buffer before filling it up again
    flush();

    while (local_time >= m_bufferTimesteps) {
      m_timeStartThisBatch += m_bufferTimesteps;
      local_time = timeIndex - m_timeStartThisBatch;
    }
    m_timeStartNextBatch = m_timeStartThisBatch + m_bufferTimesteps;
  }

  // Keep track of all time indices pushed into the buffer
  m_timeindices.insert(timeIndex);

  // Copy data into buffers
  m_bufferUVW(local_bl, local_time) = {static_cast<float>(uvwInMeters[0]),
                                       static_cast<float>(uvwInMeters[1]),
                                       static_cast<float>(uvwInMeters[2])};

  m_bufferStationPairs(local_bl) = {static_cast<int>(antenna1),
                                    static_cast<int>(antenna2)};

  int nr_correlations = m_bufferset.get_nr_correlations();

  if (nr_correlations == 2) {
    for (int i = 0; i < m_nr_channels; ++i) {
      const int nr_correlations_in = 4;
      m_bufferVisibilities(local_bl, local_time, i, 0) =
          visibilities[i * nr_correlations_in];
      m_bufferVisibilities(local_bl, local_time, i, 1) =
          visibilities[i * nr_correlations_in + 3];
    }
  } else {
    std::copy_n(visibilities, m_nr_channels * nr_correlations,
                reinterpret_cast<std::complex<float>*>(
                    &m_bufferVisibilities(local_bl, local_time, 0, 0)));
  }
  int nr_correlations_weights = 4;
  std::copy_n(weights, m_nr_channels * nr_correlations_weights,
              &m_buffer_weights(local_bl, local_time, 0, 0));
}

void GridderBufferImpl::compute_avg_beam() {
  m_bufferset.get_watch(BufferSetImpl::Watch::kAvgBeam).Start();

  const size_t subgrid_size = m_bufferset.get_subgridsize();
  const size_t nr_aterms = m_aterm_offsets2.size() - 1;
  const size_t nr_antennas = m_nrStations;

  // average beam is always computed for all polarizations (for now)
  const size_t nr_correlations = 4;

  const std::array<size_t, 4> average_beam_shape{
      subgrid_size, subgrid_size, nr_correlations, nr_correlations};
  const std::array<size_t, 4> weights_shape{m_nr_baselines, m_bufferTimesteps,
                                            m_nr_channels, nr_correlations};
  const std::array<size_t, 2> uvw_shape{m_nr_baselines, m_bufferTimesteps};
  const std::array<size_t, 1> station_pairs_shape{m_nr_baselines};
  const std::array<size_t, 4> aterms_shape{nr_aterms, nr_antennas, subgrid_size,
                                           subgrid_size};
  const std::array<size_t, 1> aterm_offsets_shape{nr_aterms + 1};
  auto uvw = aocommon::xt::CreateSpan(m_bufferUVW2.data(), uvw_shape);
  auto station_pairs = aocommon::xt::CreateSpan(m_bufferStationPairs2.data(),
                                                station_pairs_shape);
  auto aterms = aocommon::xt::CreateSpan(m_aterms2.data(), aterms_shape);
  auto aterm_offsets =
      aocommon::xt::CreateSpan(m_aterm_offsets2.data(), aterm_offsets_shape);
  auto average_beam =
      aocommon::xt::CreateSpan(m_average_beam, average_beam_shape);
  auto weights =
      aocommon::xt::CreateSpan(m_buffer_weights2.data(), weights_shape);

  proxy::Proxy& proxy = m_bufferset.get_proxy();
  proxy.compute_avg_beam(m_nrStations, get_frequencies_size(), uvw,
                         station_pairs, aterms, aterm_offsets, weights,
                         average_beam);

  m_bufferset.get_watch(BufferSetImpl::Watch::kAvgBeam).Pause();

}  // end compute_avg_beam

void GridderBufferImpl::flush_thread_worker() {
  if (m_average_beam) {
    compute_avg_beam();
  }

  if (!m_bufferset.get_do_gridding()) return;

  const size_t subgridsize = m_bufferset.get_subgridsize();

  auto aterm_offsets_span =
      m_bufferset.get_apply_aterm()
          ? aocommon::xt::CreateSpan<unsigned int, 1>(m_aterm_offsets2.data(),
                                                      {m_aterm_offsets2.size()})
          : aocommon::xt::CreateSpan<unsigned int, 1>(
                m_default_aterm_offsets.data(),
                {m_default_aterm_offsets.size()});

  auto aterms_span =
      m_bufferset.get_apply_aterm()
          ? aocommon::xt::CreateSpan<Matrix2x2<std::complex<float>>, 4>(
                m_aterms2.data(), {m_aterm_offsets2.size() - 1, m_nrStations,
                                   subgridsize, subgridsize})
          : aocommon::xt::CreateSpan<Matrix2x2<std::complex<float>>, 4>(
                m_default_aterms.data(),
                {m_default_aterm_offsets.size() - 1, m_nrStations, subgridsize,
                 subgridsize});

  proxy::Proxy& proxy = m_bufferset.get_proxy();

  // Set Plan options
  Plan::Options options;
  options.nr_w_layers = proxy.get_grid().shape(0);
  options.plan_strict = false;
  options.mode = (m_bufferset.get_nr_polarizations() == 4)
                     ? Plan::Mode::FULL_POLARIZATION
                     : Plan::Mode::STOKES_I_ONLY;

  // Create plan
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Start();
  std::unique_ptr<Plan> plan = proxy.make_plan(
      m_bufferset.get_kernel_size(), m_frequencies, m_bufferUVW2,
      m_bufferStationPairs2, aterm_offsets_span, options);
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Pause();

  // Run gridding
  m_bufferset.get_watch(BufferSetImpl::Watch::kGridding).Start();
  proxy.gridding(*plan, m_frequencies, m_bufferVisibilities2, m_bufferUVW2,
                 m_bufferStationPairs2, aterms_span, aterm_offsets_span,
                 m_bufferset.get_taper());
  m_bufferset.get_watch(BufferSetImpl::Watch::kGridding).Pause();
}

// Must be called whenever the buffer is full or no more data added
void GridderBufferImpl::flush() {
  // Return if no input in buffer
  if (m_timeindices.size() == 0) return;

  // if there is still a flushthread running, wait for it to finish
  if (m_flush_thread.joinable()) m_flush_thread.join();

  const size_t subgridsize = m_bufferset.get_subgridsize();

  std::swap(m_bufferUVW, m_bufferUVW2);
  std::swap(m_bufferStationPairs, m_bufferStationPairs2);
  std::swap(m_bufferVisibilities, m_bufferVisibilities2);
  std::swap(m_buffer_weights, m_buffer_weights2);
  std::swap(m_aterm_offsets, m_aterm_offsets2);

  std::swap(m_aterms, m_aterms2);

  m_flush_thread = std::thread(&GridderBufferImpl::flush_thread_worker, this);

  // Prepare next batch
  m_timeStartThisBatch += m_bufferTimesteps;
  m_timeStartNextBatch += m_bufferTimesteps;
  m_timeindices.clear();
  reset_aterm();
  set_uvw_to_infinity();
}

// Reset the a-term for a new buffer; copy the last a-term from the
// previous buffer;
void GridderBufferImpl::reset_aterm() {
  if (m_aterm_offsets.size() != 2) {
    m_aterm_offsets = std::vector<unsigned int>(2, 0);
  }
  m_aterm_offsets[0] = 0;
  m_aterm_offsets[1] = m_bufferTimesteps;

  size_t n_old_aterms =
      m_aterm_offsets2.size() - 1;  // Nr aterms in previous chunk

  const size_t subgridsize = m_bufferset.get_subgridsize();
  size_t atermBlockSize = m_nrStations * subgridsize * subgridsize;
  m_aterms.resize(atermBlockSize);
  std::copy_n(m_aterms2.data() + (n_old_aterms - 1) * atermBlockSize,
              atermBlockSize, m_aterms.data());
}

void GridderBufferImpl::finished() {
  flush();
  // if there is still a flushthread running, wait for it to finish
  if (m_flush_thread.joinable()) {
    m_flush_thread.join();
  }

  // Retrieve the grid, this makes sure that any operations in the proxy
  // (e.g.) w-tiling, is finished and the grid passed in ::flush() can
  // be used again by the caller.
  proxy::Proxy& proxy = m_bufferset.get_proxy();
  proxy.get_final_grid();
}

void GridderBufferImpl::malloc_buffers() {
  BufferImpl::malloc_buffers();

  const size_t nr_correlations = m_bufferset.get_nr_correlations();
  proxy::Proxy& proxy = m_bufferset.get_proxy();
  m_bufferUVW2 =
      proxy.allocate_span<UVW<float>, 2>({m_nr_baselines, m_bufferTimesteps});
  m_bufferVisibilities2 = proxy.allocate_span<std::complex<float>, 4>(
      {m_nr_baselines, m_bufferTimesteps, m_nr_channels, nr_correlations});
  m_bufferStationPairs2 =
      proxy.allocate_span<std::pair<unsigned int, unsigned int>, 1>(
          {m_nr_baselines});
  const size_t nr_correlations_weights = 4;
  m_buffer_weights =
      proxy.allocate_span<float, 4>({m_nr_baselines, m_bufferTimesteps,
                                     m_nr_channels, nr_correlations_weights});
  m_buffer_weights2 =
      proxy.allocate_span<float, 4>({m_nr_baselines, m_bufferTimesteps,
                                     m_nr_channels, nr_correlations_weights});
}

}  // namespace api
}  // namespace idg
