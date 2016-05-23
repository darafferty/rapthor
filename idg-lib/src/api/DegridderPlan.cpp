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


    void DegridderPlan::load_visibilities(size_t rowId,
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
        auto largestTimeIndex = *max_element( m_timeindices.cbegin(), m_timeindices.cend() );
        m_lastTimeIndex = largestTimeIndex;
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


    int DegridderPlan_get_stations(idg::DegridderPlan* p)
    {
        return p->get_stations();
    }


    void DegridderPlan_set_stations(idg::DegridderPlan* p, int n) {
        p->set_stations(n);
    }


    void DegridderPlan_set_frequencies(
        idg::DegridderPlan* p,
        double* frequencyList,
        int size)
    {
        p->set_frequencies(frequencyList, size);
    }


    double DegridderPlan_get_frequency(idg::DegridderPlan* p, int channel)
    {
        return p->get_frequency(channel);
    }


    int DegridderPlan_get_frequencies_size(idg::DegridderPlan* p)
    {
        return p->get_frequencies_size();
    }


    void DegridderPlan_set_w_kernel_size(idg::DegridderPlan* p, int size)
    {
        p->set_w_kernel(size);
    }


    int DegridderPlan_get_w_kernel_size(idg::DegridderPlan* p)
    {
        return p->get_w_kernel_size();
    }



    void DegridderPlan_set_grid(
        idg::DegridderPlan* p,
        void* grid,   // ptr to complex double
        int nr_polarizations,
        int height,
        int width
        )
    {
        p->set_grid(
            (std::complex<double>*) grid,
            nr_polarizations,
            height,
            width);
    }


    void DegridderPlan_set_spheroidal(
        idg::DegridderPlan* p,
        double* spheroidal,
        int height,
        int width)
    {
        p->set_spheroidal(spheroidal, height, width);
    }



    // deprecated: use cell size!
    void DegridderPlan_set_image_size(idg::DegridderPlan* p, double imageSize)
    {
        p->set_image_size(imageSize);
    }


    // deprecated: use cell size!
    double DegridderPlan_get_image_size(idg::DegridderPlan* p)
    {
        return p->get_image_size();
    }


    void DegridderPlan_bake(idg::DegridderPlan* p)
    {
        p->bake();
    }


    void DegridderPlan_start_aterm(
        idg::DegridderPlan* p,
        void* aterm,  // ptr to complex double
        int nrStations,
        int height,
        int width,
        int nrPolarizations)
    {
        p->start_aterm(
            (std::complex<double>*) aterm,
            nrStations,
            height,
            width,
            nrPolarizations);
    }


    void DegridderPlan_finish_aterm(idg::DegridderPlan* p)
    {
        p->finish_aterm();
    }


    void DegridderPlan_flush(idg::DegridderPlan* p)
    {
        p->flush();
    }


    void DegridderPlan_internal_set_subgrid_size(idg::DegridderPlan* p, int size)
    {
        p->internal_set_subgrid_size(size);
    }


    int DegridderPlan_internal_get_subgrid_size(idg::DegridderPlan* p)
    {
        return p->internal_get_subgrid_size();
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


    void DegridderPlan_load_visibilities(idg::DegridderPlan* p,
                                         int rowId,
                                         void* visibilities) // ptr to complex<float>
    {
        p->load_visibilities(rowId, (std::complex<float>*) visibilities);
    }


} // extern C
