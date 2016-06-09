/*
 * DegridderPlan.h
 * Access to IDG's high level degridder routines
 */


#include "DegridderPlan.h"


using namespace std;

namespace idg {

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


    void DegridderPlan::request_visibilities(
        const double* uvwInMeters,
        size_t antenna1,
        size_t antenna2,
        size_t timeIndex)
    {
        // exclude auto-correlations
        if (antenna1 == antenna2) return;

        int    local_time = timeIndex - m_timeStartThisBatch;
        size_t local_bl   = baseline_index(antenna1, antenna2);

        // In this API it is the responsibility of the user to read the buffer
        // before requesting visibilities outside of the current time window.
        // If time index outside of the window is requested we reset the buffers
        // and assume a new time window is processed from now on.
        if (local_time < 0) {
            m_timeStartThisBatch = 0;
            m_timeStartNextBatch = m_bufferTimesteps;
            local_time = timeIndex;
        }

        if (local_time >= m_bufferTimesteps) {
            while (local_time >= m_bufferTimesteps) {
                m_timeStartThisBatch += m_bufferTimesteps;
                local_time = timeIndex - m_timeStartThisBatch;
            }
            m_timeStartNextBatch = m_timeStartThisBatch + m_bufferTimesteps;
        }

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
    }


    void DegridderPlan::read_visibilities(size_t antenna1,
                                          size_t antenna2,
                                          size_t timeIndex,
                                          complex<float>* visibilities)
    {
        // exclude auto-correlations
        if (antenna1 == antenna2)
            throw invalid_argument("Cannot read if Antenna 1 == Antenna 2");

        // Make sure visibilities are computed before read
        flush();

        // Read from buffer: note m_timeStartThisBatch is fro the next batch at this point
        int    local_time = timeIndex - (m_timeStartThisBatch - m_bufferTimesteps);
        size_t local_bl   = baseline_index(antenna1, antenna2);

        if (local_time < 0 || local_time >= m_bufferTimesteps)
            throw invalid_argument("Attempt to read visibilites not in local buffer.");

        complex<float>* start_ptr = (complex<float>*)
            &m_bufferVisibilities(local_bl, local_time, 0);
        copy(start_ptr, start_ptr + get_frequencies_size() * m_nrPolarizations,
             visibilities);
    }


    // Must be called whenever the buffer is full or no more data added
    void DegridderPlan::flush()
    {
        // Return if no input in buffer
        if (m_timeindices.size() == 0) return;

        // TODO: remove below //////////////////////////
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
        // TODO: remove above //////////////////////////

        // int kernSize = max(m_wKernelSize, m_aKernelSize, m_spheroidalKernelSize);
        int kernelSize = m_wKernelSize;

        m_proxy->degrid_visibilities(
            (complex<float>*) m_bufferVisibilities.data(),
            (float*) m_bufferUVW.data(),
            (float*) m_wavenumbers.data(),
            (int*) m_bufferStationPairs.data(),
            (complex<float>*) m_grid.data(),
            m_wOffsetInLambda,
            kernelSize,
            (complex<float>*) m_aterms.data(),
            m_aterm_offsets,
            (float*) m_spheroidal.data());

        // Prepare next batch
        m_timeStartThisBatch += m_bufferTimesteps;
        m_timeStartNextBatch += m_bufferTimesteps;
        m_timeindices.clear();
        set_uvw_to_infinity();
    }


    void DegridderPlan::transform_grid(complex<double> *grid)
    {
        fft_grid(grid);
        // apply spheroidal
    }

} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    idg::DegridderPlan* DegridderPlan_init(unsigned int type,
                                           unsigned int bufferTimesteps)
    {
        auto proxytype = idg::Type::CPU_REFERENCE;
        if (type == 0) {
            proxytype = idg::Type::CPU_REFERENCE;
        } else if (type == 1) {
            proxytype = idg::Type::CPU_OPTIMIZED;
        }
        return new idg::DegridderPlan(proxytype, bufferTimesteps);
    }

    void DegridderPlan_destroy(idg::DegridderPlan* p) {
       delete p;
    }

    void DegridderPlan_request_visibilities(
        idg::DegridderPlan* p,
        double* uvwInMeters,
        int antenna1,
        int antenna2,
        int timeIndex)
    {
        p->request_visibilities(
            uvwInMeters,
            antenna1,
            antenna2,
            timeIndex);
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
            (complex<float>*) visibilities);
    }


    void DegridderPlan_transform_grid(
        idg::DegridderPlan* p,
        void *grid)
    {
        p->transform_grid((complex<double> *) grid);
    }

} // extern C
