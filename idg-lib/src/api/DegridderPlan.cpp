/*
 * DegridderPlan.h
 * Access to IDG's high level gridder routines
 */


#include "DegridderPlan.h"


using namespace std;

namespace idg {

    // Constructors and destructor
    DegridderPlan::DegridderPlan(Type architecture,
                                 size_t bufferTimesteps)
        : Scheme(architecture, bufferTimesteps)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }


    DegridderPlan::~DegridderPlan()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }

    // Degridding routines

    void DegridderPlan::request_visibilities(
        size_t rowId,
        const double* uvwInMeters,
        size_t antenna1,
        size_t antenna2,
        size_t timeIndex)
    {
        size_t local_time = timeIndex - m_lastTimeIndex - 1;
        size_t local_bl = baseline_index(antenna1, antenna2);

        if (local_time >= m_bufferTimesteps) {
            /* Do not insert more if buffer is already full */
            /* Empty buffer, before inserting new element   */
            flush();
            local_time = 0;
        } else {
            // Keep track of all time indices pushed into the buffer
            m_timeindices.insert(timeIndex);

            // Copy data into buffers
            m_bufferUVW(local_bl, local_time) = {
                static_cast<float>(uvwInMeters[0]),
                static_cast<float>(uvwInMeters[1]),
                static_cast<float>(uvwInMeters[2])
            };

            if (antenna1 > antenna2) swap(antenna1, antenna2);
            m_bufferStationPairs[local_bl] = {
                int(antenna1),
                int(antenna2)
            };

            m_rowid_to_bufferindex.emplace(
                make_pair(rowId, make_pair(local_bl, local_time)));
        }
    }


    void DegridderPlan::read_visibilities(size_t antenna1,
                                          size_t antenna2,
                                          size_t timeIndex,
                                          std::complex<float>* visibilities) const
    {
        size_t local_time = timeIndex - m_lastTimeIndex - 1;
        size_t local_bl = baseline_index(antenna1, antenna2);
        complex<float>* start_ptr = (complex<float>*)
            &m_bufferVisibilities(local_bl, local_time, 0);
        copy(start_ptr, start_ptr + get_frequencies_size() * m_nrPolarizations,
             visibilities);
        // m_rowid_to_bufferindex.remove(rowId);
    }


    void DegridderPlan::read_visibilities(size_t rowId,
                                          std::complex<float>* visibilities) const
    {
        pair<size_t,size_t> indices = m_rowid_to_bufferindex.at(rowId);
        size_t local_bl   = indices.first;
        size_t local_time = indices.second;
        complex<float>* start_ptr = (complex<float>*)
            &m_bufferVisibilities(local_bl, local_time, 0);
        copy(start_ptr, start_ptr + get_frequencies_size() * m_nrPolarizations,
             visibilities);
        // m_rowid_to_bufferindex.remove(rowId);
    }


    // Must be called whenever the buffer is full or no more data added
    void DegridderPlan::flush()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        if (m_timeindices.size() == 0) return;

        // HACK: copy double precison grid to single precison
        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    m_grid(p, y, x) = complex<float>(
                        m_grid_double[p*m_gridHeight*m_gridWidth
                                      + y*m_gridWidth + x]);
                }
            }
        }

        int kernelsize = m_wKernelSize;

        // TODO: this routine should be not much more than this call
        m_proxy->degrid_visibilities(
            (complex<float>*) m_bufferVisibilities.data(),
            (float*) m_bufferUVW.data(),
            (float*) m_wavenumbers.data(),
            (int*) m_bufferStationPairs.data(),
            (complex<float>*) m_grid.data(),
            m_wOffsetInMeters,
            kernelsize,
            (complex<float>*) m_aterms.data(),
            m_aterm_offsets,
            (float*) m_spheroidal.data());

        // Cleanup
        // auto largestTimeIndex = *max_element( m_timeindices.cbegin(), m_timeindices.cend() );
        // m_lastTimeIndex = largestTimeIndex;
        m_timeindices.clear();
        // NOT HERE: m_rowid_to_bufferindex.clear();
        // init buffers to zero
    }

} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    idg::DegridderPlan* DegridderPlan_init(unsigned int bufferTimesteps)
    {
        return new idg::DegridderPlan(idg::Type::CPU_REFERENCE, bufferTimesteps);
    }


    void DegridderPlan_destroy(idg::DegridderPlan* p) {
       delete p;
    }


    void DegridderPlan_request_visibilities(
        idg::DegridderPlan* p,
        int rowId,
        double* uvwInMeters,
        int antenna1,
        int antenna2,
        int timeIndex)
    {
        p->request_visibilities(
            rowId,
            uvwInMeters,
            antenna1,
            antenna2,
            timeIndex);
    }


    void DegridderPlan_read_visibilities_by_row_id(
        idg::DegridderPlan* p,
        int rowId,
        void* visibilities) // ptr to complex<float>
    {
        p->read_visibilities(rowId, (std::complex<float>*) visibilities);
    }


    void DegridderPlan_read_visibilities(
        idg::DegridderPlan* p,
        int antenna1,
        int antenna2,
        int timeIndex,
        void* visibilities) // ptr to complex<float>
    {
        p->read_visibilities(
            antenna1,
            antenna2,
            timeIndex,
            (std::complex<float>*) visibilities);
    }


} // extern C
