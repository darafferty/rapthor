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
      m_data_read(true) {
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
      rowId, reinterpret_cast<std::complex<float>*>(
                 m_bufferVisibilities.data(local_bl, local_time)));

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

  m_aterm_offsets_array =
      Array1D<unsigned int>(m_aterm_offsets.data(), m_aterm_offsets.size());
  m_aterms_array = Array4D<Matrix2x2<std::complex<float>>>(
      m_aterms.data(), m_aterm_offsets_array.get_x_dim() - 1, m_nrStations,
      subgridsize, subgridsize);

  // Set Plan options
  Plan::Options options;
  options.w_step = m_bufferset.get_w_step();
  options.nr_w_layers = m_bufferset.get_grid()->get_w_dim();
  options.plan_strict = false;

  proxy::Proxy& proxy = m_bufferset.get_proxy();

  // Create plan
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Start();
  std::unique_ptr<Plan> plan =
      proxy.make_plan(m_bufferset.get_kernel_size(), subgridsize,
                      m_bufferset.get_grid()->get_x_dim(),
                      m_bufferset.get_cell_size(), m_frequencies, m_bufferUVW,
                      m_bufferStationPairs, m_aterm_offsets_array, options);
  m_bufferset.get_watch(BufferSetImpl::Watch::kPlan).Pause();

  // Run degridding
  m_bufferset.get_watch(BufferSetImpl::Watch::kDegridding).Start();
  proxy.degridding(*plan, m_bufferset.get_w_step(), m_shift,
                   m_bufferset.get_cell_size(), m_bufferset.get_kernel_size(),
                   subgridsize, m_frequencies, m_bufferVisibilities,
                   m_bufferUVW, m_bufferStationPairs, *m_bufferset.get_grid(),
                   m_aterms_array, m_aterm_offsets_array,
                   m_bufferset.get_spheroidal());
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

void DegridderBufferImpl::malloc_buffers() { BufferImpl::malloc_buffers(); }

}  // namespace api
}  // namespace idg
