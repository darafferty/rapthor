/*
 * DegridderBuffer.cpp
 */

#include "DegridderBufferImpl.h"

#include <mutex>

using namespace std;

namespace idg {
namespace api {

    DegridderBufferImpl::DegridderBufferImpl(
        BufferSetImpl *bufferset,
        proxy::Proxy* proxy,
        size_t bufferTimesteps)
        : BufferImpl(bufferset, proxy, bufferTimesteps),
          m_bufferVisibilities2(0,0,0),
          m_buffer_full(false),
          m_data_read(true)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }


    DegridderBufferImpl::~DegridderBufferImpl()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }


    bool DegridderBufferImpl::request_visibilities(
        size_t rowId,
        size_t timeIndex,
        size_t antenna1,
        size_t antenna2,
        const double* uvwInMeters)
    {
        // Do not do anything if the buffer is already full
        if (m_buffer_full == true) return m_buffer_full;

        // exclude auto-correlations
        if (antenna1 == antenna2) return m_buffer_full;

        int    local_time = timeIndex - m_timeStartThisBatch;
        size_t local_bl   = baseline_index(antenna1, antenna2);

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
        m_row_ids_to_data.push_back(make_pair(rowId, (complex<float>*) &m_bufferVisibilities2(local_bl, local_time, 0)));

        // Copy data into buffers
        m_bufferUVW(local_bl, local_time) = {
            static_cast<float>(uvwInMeters[0]),
            static_cast<float>(uvwInMeters[1]),
            static_cast<float>(uvwInMeters[2])
        };

        m_bufferStationPairs(local_bl) = {
            int(antenna1),
            int(antenna2)
        };

        return m_buffer_full;
    }

    // Must be called whenever the buffer is full or no more data added
    void DegridderBufferImpl::flush()
    {
        if (m_buffer_full == true && m_data_read == false) return;

        // Return if no input in buffer
        if (m_timeindices.size() == 0) return;

        m_aterm_offsets_array = Array1D<unsigned int>(m_aterm_offsets.data(), m_aterm_offsets.size());
        m_aterms_array = Array4D<Matrix2x2<complex<float>>>(m_aterms.data(), m_aterm_offsets_array.get_x_dim()-1, m_nrStations, m_subgridsize, m_subgridsize);

        // Set Plan options
        Plan::Options options;
        options.w_step      = m_wStepInLambda;
        options.nr_w_layers = m_nr_w_layers;
        options.plan_strict = false;

        // Iterate all channel groups
        const int nr_channel_groups = m_channel_groups.size();
        for (int i = 0; i < nr_channel_groups; i++) {
            #ifndef NDEBUG
            std::cout << "degridding channels: " << m_channel_groups[i].first << "-"
                                                 << m_channel_groups[i].second << std::endl;
            #endif

            // Create plan
            Plan plan(
                m_kernel_size,
                m_subgridsize,
                m_gridHeight,
                m_cellHeight,
                m_grouped_frequencies[i],
                m_bufferUVW,
                m_bufferStationPairs,
                m_aterm_offsets_array,
                options);

            // Initialize degridding for first channel group
            if (i == 0) {
                m_proxy->initialize(
                    plan,
                    m_wStepInLambda,
                    m_shift,
                    m_cellHeight,
                    m_kernel_size,
                    m_subgridsize,
                    m_grouped_frequencies[i],
                    m_bufferVisibilities[i],
                    m_bufferUVW,
                    m_bufferStationPairs,
                    *m_grid,
                    m_aterms_array,
                    m_aterm_offsets_array,
                    m_spheroidal);
            }

            // Start flush
            m_proxy->run_degridding(
                plan,
                m_wStepInLambda,
                m_shift,
                m_cellHeight,
                m_kernel_size,
                m_subgridsize,
                m_grouped_frequencies[i],
                m_bufferVisibilities[i],
                m_bufferUVW,
                m_bufferStationPairs,
                *m_grid,
                m_aterms_array,
                m_aterm_offsets_array,
                m_spheroidal);

            // Copy data from per channel buffer into buffer for all channels
            for (int bl = 0; bl < m_nr_baselines; bl++) {
                for (int time_idx = 0;  time_idx < m_bufferTimesteps; time_idx++) {
                    std::copy(&m_bufferVisibilities[i](bl, time_idx, 0),
                            &m_bufferVisibilities[i](bl, time_idx, m_channel_groups[i].second - m_channel_groups[i].first),
                            &m_bufferVisibilities2(bl, time_idx, m_channel_groups[i].first));
                }
            }
        } // end for i (channel groups)

        // Wait for all plans to be executed
        m_proxy->finish_degridding();

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
    void DegridderBufferImpl::reset_aterm()
    {
      size_t n_old_aterms = m_aterm_offsets.size()-1; // Nr aterms in previous chunk

      if (m_aterm_offsets.size()!=2) {
        m_aterm_offsets = std::vector<unsigned int>(2, 0);
      }
      m_aterm_offsets[0] = 0;
      m_aterm_offsets[1] = m_bufferTimesteps;

      size_t atermBlockSize = m_nrStations*m_subgridsize*m_subgridsize;
      std::copy(m_aterms.data()+(n_old_aterms-1)*atermBlockSize,
                m_aterms.data()+(n_old_aterms)*atermBlockSize,
                (Matrix2x2<complex<float>>*) m_aterms.data());
      m_aterms.resize(atermBlockSize);
    }

    std::vector<std::pair<size_t, std::complex<float>*>> DegridderBufferImpl::compute()
    {
        flush();
        m_buffer_full =  false;
        return std::move(m_row_ids_to_data);
    }


    void DegridderBufferImpl::finished_reading()
    {
        #if defined(DEBUG)
        cout << "FINISHED READING: buffer full " << m_buffer_full << endl;
        cout << "m_row_ids_to_data size: " << m_row_ids_to_data.size() << endl;
        #endif
        m_row_ids_to_data.clear();
        m_data_read = true;
    }

    void DegridderBufferImpl::malloc_buffers()
    {
        BufferImpl::malloc_buffers();
        m_bufferVisibilities2 = Array3D<Visibility<std::complex<float>>>(m_nr_baselines, m_bufferTimesteps, get_frequencies_size());
    }

} // namespace api
} // namespace idg
