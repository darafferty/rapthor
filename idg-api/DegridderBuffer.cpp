// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * DegridderBuffer.cpp
 */

#include "DegridderBufferImpl.h"
#include "BufferSetImpl.h"

#include <algorithm>
#include <csignal>

namespace idg {
namespace api {

DegridderBufferImpl::DegridderBufferImpl(const BufferSetImpl& bufferset,
                                         size_t bufferTimesteps)
    : BufferImpl(bufferset, bufferTimesteps),
      m_buffer_full(false),
      m_data_read(true),
      m_bufferVisibilities2(aocommon::xt::CreateSpan<std::complex<float>, 4>(
          nullptr, {0, 0, 0, 0})) {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif
}

DegridderBufferImpl::~DegridderBufferImpl() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif
}

bool DegridderBufferImpl::request_visibilities(size_t rowId, size_t timeIndex,
                                               size_t antenna1, size_t antenna2,
                                               const double* uvwInMeters) {
  // Do not do anything if the buffer is already full
  if (m_buffer_full == true) return m_buffer_full;

  // exclude auto-correlations
  if (antenna1 == antenna2) return m_buffer_full;

  int local_time = timeIndex - m_timeStartThisBatch;
  size_t local_bl = Plan::baseline_index(antenna1, antenna2, m_nrStations);

  // #if defined(DEBUG)
  // cout << "REQUEST: row " << rowId << ", local time " << local_time << endl;
  // #endif

  if (local_time < 0) {
    m_buffer_full = false;
    m_timeStartThisBatch = 0;
    m_timeStartNextBatch = m_bufferTimesteps;
    local_time = timeIndex;
  }

  if (local_time >= m_bufferTimesteps) {
    m_buffer_full = true;

    while (local_time > m_bufferTimesteps) {
      m_timeStartThisBatch += m_bufferTimesteps;
      local_time = timeIndex - m_timeStartThisBatch;
    }
    m_timeStartNextBatch = m_timeStartThisBatch + m_bufferTimesteps;

    return m_buffer_full;
  }

  // Keep track of all time indices pushed into the buffer
  m_timeindices.insert(timeIndex);

  // #if defined(DEBUG)
  // cout << "INSERT: {" << rowId << ", (" << local_bl << ", "
  //      << local_time << ") }" << endl;
  // #endif

  // Keep mapping rowId -> (local_bl, local_time) for reading
  m_row_ids_to_data.emplace_back(
      rowId, &m_bufferVisibilities(local_bl, local_time, 0, 0));

  // Copy data into buffers
  m_bufferUVW(local_bl, local_time) = {static_cast<float>(uvwInMeters[0]),
                                       static_cast<float>(uvwInMeters[1]),
                                       static_cast<float>(uvwInMeters[2])};

  m_bufferStationPairs(local_bl) = {int(antenna1), int(antenna2)};

  return m_buffer_full;
}

// Must be called whenever the buffer is full or no more data added
void DegridderBufferImpl::flush() {
  if (m_buffer_full == true && m_data_read == false) return;

  // Return if no input in buffer
  if (m_timeindices.size() == 0) return;

  const size_t subgridsize = m_bufferset.get_subgridsize();

  auto aterm_offsets_span = aocommon::xt::CreateSpan<unsigned int, 1>(
      m_aterm_offsets.data(), {m_aterm_offsets.size()});
  auto aterms_span =
      aocommon::xt::CreateSpan<Matrix2x2<std::complex<float>>, 4>(
          m_aterms.data(),
          {m_aterm_offsets.size() - 1, m_nrStations, subgridsize, subgridsize});

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
  std::unique_ptr<Plan> plan =
      proxy.make_plan(m_bufferset.get_kernel_size(), m_frequencies, m_bufferUVW,
                      m_bufferStationPairs, aterm_offsets_span, options);
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Pause();

  // Run degridding
  m_bufferset.get_watch(BufferSetImpl::Watch::kDegridding).Start();
  proxy.degridding(*plan, m_frequencies, m_bufferVisibilities, m_bufferUVW,
                   m_bufferStationPairs, aterms_span, aterm_offsets_span,
                   m_bufferset.get_taper());
  m_bufferset.get_watch(BufferSetImpl::Watch::kDegridding).Pause();

  // Prepare next batch
  m_timeStartThisBatch += m_bufferTimesteps;
  m_timeStartNextBatch += m_bufferTimesteps;
  m_timeindices.clear();

  set_uvw_to_infinity();
  reset_aterm();

  m_data_read = false;
}

// Reset the a-term for a new buffer; copy the last a-term from the
// previous buffer;
void DegridderBufferImpl::reset_aterm() {
  size_t n_old_aterms =
      m_aterm_offsets.size() - 1;  // Nr aterms in previous chunk

  if (m_aterm_offsets.size() != 2) {
    m_aterm_offsets = std::vector<unsigned int>(2, 0);
  }
  m_aterm_offsets[0] = 0;
  m_aterm_offsets[1] = m_bufferTimesteps;

  const size_t subgridsize = m_bufferset.get_subgridsize();
  size_t atermBlockSize = m_nrStations * subgridsize * subgridsize;
  std::copy(m_aterms.data() + (n_old_aterms - 1) * atermBlockSize,
            m_aterms.data() + (n_old_aterms)*atermBlockSize, m_aterms.data());
  m_aterms.resize(atermBlockSize);
}

std::vector<std::pair<size_t, std::complex<float>*>>
DegridderBufferImpl::compute() {
  flush();
  m_buffer_full = false;
  if (m_bufferset.get_nr_correlations() == 2) {
    m_bufferVisibilities2.fill(std::complex<float>(0.0f, 0.0f));
    for (size_t row_id = 0; row_id < m_row_ids_to_data.size(); ++row_id) {
      const size_t bl = row_id / m_bufferTimesteps;
      const size_t time = row_id % m_bufferTimesteps;
      for (size_t chan = 0; chan < m_nr_channels; ++chan) {
        m_bufferVisibilities2(bl, time, chan, 0) =
            *(m_row_ids_to_data[row_id].second + chan * 2);
        m_bufferVisibilities2(bl, time, chan, 3) =
            *(m_row_ids_to_data[row_id].second + chan * 2 + 1);
      }
      m_row_ids_to_data[row_id].second = &m_bufferVisibilities2(bl, time, 0, 0);
    }
  }
  return std::move(m_row_ids_to_data);
}

void DegridderBufferImpl::finished_reading() {
#if defined(DEBUG)
  cout << "FINISHED READING: buffer full " << m_buffer_full << endl;
  cout << "m_row_ids_to_data size: " << m_row_ids_to_data.size() << endl;
#endif
  m_row_ids_to_data.clear();
  m_data_read = true;
}

void DegridderBufferImpl::malloc_buffers() {
  BufferImpl::malloc_buffers();

  if (m_bufferset.get_nr_correlations() == 2) {
    m_bufferVisibilities2 =
        m_bufferset.get_proxy().allocate_span<std::complex<float>, 4>(
            {m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4});
  }
}

}  // namespace api
}  // namespace idg
