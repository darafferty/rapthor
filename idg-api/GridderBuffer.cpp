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

GridderBufferImpl::GridderBufferImpl(const BufferSetImpl &bufferset,
                                     size_t bufferTimesteps)
    : BufferImpl(bufferset, bufferTimesteps),
      m_bufferUVW2(0, 0),
      m_bufferStationPairs2(0),
      m_buffer_weights(0, 0, 0, 0),
      m_buffer_weights2(0, 0, 0, 0),
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

void GridderBufferImpl::grid_visibilities(size_t timeIndex, size_t antenna1,
                                          size_t antenna2,
                                          const double *uvwInMeters,
                                          std::complex<float> *visibilities,
                                          const float *weights) {
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

  std::copy_n(visibilities, m_nr_channels * m_nrPolarizations,
              reinterpret_cast<std::complex<float> *>(
                  &m_bufferVisibilities(local_bl, local_time, 0)));
  std::copy_n(weights, m_nr_channels * 4,
              &m_buffer_weights(local_bl, local_time, 0, 0));
}

void GridderBufferImpl::compute_avg_beam() {
  m_bufferset.get_watch(BufferSetImpl::Watch::kAvgBeam).Start();

  const unsigned int subgrid_size = m_bufferset.get_subgridsize();
  const unsigned int nr_correlations = 4;
  const unsigned int nr_aterms = m_aterm_offsets2.size() - 1;
  const unsigned int nr_antennas = m_nrStations;
  const unsigned int nr_baselines = m_bufferStationPairs2.get_x_dim();
  const unsigned int nr_timesteps = m_bufferUVW2.get_x_dim();
  const unsigned int nr_channels = get_frequencies_size();

  Array4D<Matrix2x2<std::complex<float>>> aterms(
      m_aterms2.data(), nr_aterms, nr_antennas, subgrid_size, subgrid_size);
  Array1D<unsigned int> aterms_offsets(m_aterm_offsets2.data(), nr_aterms + 1);
  idg::Array4D<std::complex<float>> average_beam(m_average_beam, subgrid_size,
                                                 subgrid_size, 4, 4);

  proxy::Proxy &proxy = m_bufferset.get_proxy();
  proxy.compute_avg_beam(m_nrStations, get_frequencies_size(), m_bufferUVW2,
                         m_bufferStationPairs2, aterms, aterms_offsets,
                         m_buffer_weights2, average_beam);

  m_bufferset.get_watch(BufferSetImpl::Watch::kAvgBeam).Pause();

}  // end compute_avg_beam

void GridderBufferImpl::flush_thread_worker() {
  if (m_average_beam) {
    compute_avg_beam();
  }

  if (!m_bufferset.get_do_gridding()) return;

  const size_t subgridsize = m_bufferset.get_subgridsize();

  const Array4D<std::complex<float>> *aterm_correction;
  if (m_bufferset.get_apply_aterm()) {
    aterm_correction = &m_bufferset.get_avg_aterm_correction();
  } else {
    m_aterm_offsets_array = Array1D<unsigned int>(
        m_default_aterm_offsets.data(), m_default_aterm_offsets.size());
    m_aterms_array = Array4D<Matrix2x2<std::complex<float>>>(
        m_default_aterms.data(), m_default_aterm_offsets.size() - 1,
        m_nrStations, subgridsize, subgridsize);
    aterm_correction = &m_bufferset.get_default_aterm_correction();
  }

  // Set Plan options
  Plan::Options options;
  options.w_step = m_bufferset.get_w_step();
  options.nr_w_layers = m_bufferset.get_grid()->get_w_dim();
  options.plan_strict = false;

  proxy::Proxy &proxy = m_bufferset.get_proxy();

  // Create plan
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Start();
  std::unique_ptr<Plan> plan =
      proxy.make_plan(m_bufferset.get_kernel_size(), subgridsize,
                      m_bufferset.get_grid()->get_x_dim(),
                      m_bufferset.get_cell_size(), m_frequencies, m_bufferUVW2,
                      m_bufferStationPairs2, m_aterm_offsets_array, options);
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Pause();

  // Run gridding
  m_bufferset.get_watch(BufferSetImpl::Watch::kGridding).Start();
  proxy.gridding(*plan, m_bufferset.get_w_step(), m_shift,
                 m_bufferset.get_cell_size(), m_bufferset.get_kernel_size(),
                 subgridsize, m_frequencies, m_bufferVisibilities2,
                 m_bufferUVW2, m_bufferStationPairs2, m_aterms_array,
                 m_aterm_offsets_array, m_bufferset.get_spheroidal());
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
  m_aterm_offsets_array =
      Array1D<unsigned int>(m_aterm_offsets2.data(), m_aterm_offsets2.size());

  std::swap(m_aterms, m_aterms2);
  assert(m_aterms2.size() == (m_aterm_offsets_array.get_x_dim() - 1) *
                                 m_nrStations * subgridsize * subgridsize);
  m_aterms_array = Array4D<Matrix2x2<std::complex<float>>>(
      m_aterms2.data(), m_aterm_offsets_array.get_x_dim() - 1, m_nrStations,
      subgridsize, subgridsize);

  // Pass the grid to the proxy
  proxy::Proxy &proxy = m_bufferset.get_proxy();
  proxy.set_grid(*m_bufferset.get_grid());

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
  std::copy(m_aterms2.data() + (n_old_aterms - 1) * atermBlockSize,
            m_aterms2.data() + (n_old_aterms)*atermBlockSize,
            (Matrix2x2<std::complex<float>> *)m_aterms.data());
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
  proxy::Proxy &proxy = m_bufferset.get_proxy();
  proxy.get_grid();
}

void GridderBufferImpl::malloc_buffers() {
  BufferImpl::malloc_buffers();

  proxy::Proxy &proxy = m_bufferset.get_proxy();
  m_bufferUVW2 =
      proxy.allocate_array2d<UVW<float>>(m_nr_baselines, m_bufferTimesteps);
  m_bufferVisibilities2 =
      proxy.allocate_array3d<Visibility<std::complex<float>>>(
          m_nr_baselines, m_bufferTimesteps, m_nr_channels);
  m_bufferStationPairs2 =
      proxy.allocate_array1d<std::pair<unsigned int, unsigned int>>(
          m_nr_baselines);
  m_buffer_weights = proxy.allocate_array4d<float>(
      m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4);
  m_buffer_weights2 = proxy.allocate_array4d<float>(
      m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4);
}

}  // namespace api
}  // namespace idg