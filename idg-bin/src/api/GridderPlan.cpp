/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */


#include "GridderPlan.h"


using namespace std;

namespace idg {

    // Constructors and destructor
    GridderPlan::GridderPlan(Type architecture,
                             size_t bufferTimesteps)
        : Scheme(architecture, bufferTimesteps)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }

    GridderPlan::~GridderPlan()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }

    // Gridding routines

    void GridderPlan::grid_visibilities(
        const complex<float>* visibilities, // size CH x PL
        const double* uvwInMeters,
        size_t antenna1,
        size_t antenna2,
        size_t timeIndex)
    {
        auto local_time = timeIndex - m_lastTimeIndex - 1;
        auto local_bl = baseline_index(antenna1, antenna2);

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
                static_cast<int>(antenna1),
                static_cast<int>(antenna2)
            };

            copy(visibilities, visibilities + get_frequencies_size() * m_nrPolarizations,
                 (complex<float>*) &m_bufferVisibilities(local_bl, local_time, 0));
        }
    }


    // Must be called whenever the buffer is full or no more data added
    void GridderPlan::flush()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        if (m_timeindices.size() == 0) return;

        int kernelsize = m_wKernelSize;

        // TODO: this routine should be not much more than this call
        m_proxy->grid_visibilities(
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

        // HACK: Add results to double precision grid
        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    m_grid_double[p*m_gridHeight*m_gridWidth
                                  + y*m_gridWidth + x] += m_grid(p, y, x);
                }
            }
        }

        // Cleanup
        auto largestTimeIndex = *max_element( m_timeindices.cbegin(), m_timeindices.cend() );
        m_lastTimeIndex = largestTimeIndex;
        m_timeindices.clear();
        // init buffers to zero
    }

} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    idg::GridderPlan* GridderPlan_init(unsigned int bufferTimesteps)
    {
        return new idg::GridderPlan(idg::Type::CPU_REFERENCE, bufferTimesteps);
    }


    void GridderPlan_destroy(idg::GridderPlan* p) {
       delete p;
    }


    void GridderPlan_grid_visibilities(
        idg::GridderPlan* p,
        float*  visibilities, // size CH x PL x 2
        double* uvwInMeters,
        int     antenna1,
        int     antenna2,
        int     timeIndex)
    {
        p->grid_visibilities(
            (complex<float>*) visibilities, // size CH x PL
            uvwInMeters,
            antenna1,
            antenna2,
            timeIndex);
    }

} // extern C
