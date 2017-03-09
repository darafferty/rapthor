/*
 * DegridderBuffer.cpp
 */

#include "DegridderBufferImpl.h"

using namespace std;

namespace idg {
namespace api {

    DegridderBufferImpl::DegridderBufferImpl(Type architecture,
                                 size_t bufferTimesteps)
        : BufferImpl(architecture, bufferTimesteps),
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
        m_row_ids_to_data.push_back(make_pair(rowId, (complex<float>*) &m_bufferVisibilities(local_bl, local_time, 0)));
        
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

//         // TODO: remove below //////////////////////////
//         // HACK: copy double precison grid to single precison
//         for (auto p = 0; p < m_nrPolarizations; ++p) {
//             for (auto y = 0; y < m_gridHeight; ++y) {
//                 for (auto x = 0; x < m_gridWidth; ++x) {
//                     m_grid(0, p, y, x) = complex<float>(
//                         m_grid_double[p*m_gridHeight*m_gridWidth
//                                       + y*m_gridWidth + x]);
//                 }
//             }
//         }
// 
//         // TODO: remove above //////////////////////////

        m_proxy->degridding(
            m_wStepInLambda,
            m_cellHeight,
            m_kernel_size,
            m_frequencies,
            m_bufferVisibilities,
            m_bufferUVW,
            m_bufferStationPairs,
            *m_grid,
            m_aterms,
            m_aterm_offsets,
            m_spheroidal);

        // Prepare next batch
        m_timeStartThisBatch += m_bufferTimesteps;
        m_timeStartNextBatch += m_bufferTimesteps;
        m_timeindices.clear();

        set_uvw_to_infinity();

        m_data_read = false;
    }


//     void DegridderBufferImpl::transform_grid(
//         double crop_tolerance,
//         size_t nr_polarizations,
//         size_t height,
//         size_t width,
//         complex<double> *grid)
//     {
//         // Normal case: no arguments -> transform member grid
//         // Note: the other case is to perform the transform on a copy
//         // so that the process can be monitored
//         if (grid == nullptr) {
//             nr_polarizations = m_nrPolarizations;
//             height           = m_gridHeight;
//             width            = m_gridWidth;
//             grid             = m_grid_double;
//         }
// 
//         // FFT complex-to-complex for each polarization
//         std::cout << "calling fft_grid" << std::endl;
//         fft_grid(nr_polarizations, height, width, grid);
//         std::cout << "returned from fft_grid" << std::endl;
// 
//         // // TODO: Apply spheroidal here as well?
//         // Grid2D<float> spheroidal_grid(height, width);
//         // resize2f(static_cast<int>(m_subgridSize),
//         //          static_cast<int>(m_subgridSize),
//         //          m_spheroidal.data(),
//         //          static_cast<int>(height),
//         //          static_cast<int>(width),
//         //          spheroidal_grid.data());
// 
//         // for (auto pol = 0; pol < nr_polarizations; ++pol) {
//         //     for (auto y = 0; y < height; ++y) {
//         //         for (auto x = 0; x < width; ++x) {
//         //             complex<double> scale;
//         //             if (spheroidal_grid(y,x) >= crop_tolerance) {
//         //                 scale = complex<double>(1.0/spheroidal_grid(y,x));
//         //             } else {
//         //                 scale = 0.0;
//         //             }
//         //             grid[pol*height*width + y*width + x] *= scale;
//         //         }
//         //     }
//         // }
//     }


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
        #endif
        cout << "m_row_ids_to_data size: " << m_row_ids_to_data.size() << endl;
        m_row_ids_to_data.clear();
        m_data_read = true;
    }

} // namespace api
} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    idg::api::DegridderBuffer* DegridderBuffer_init(unsigned int type,
                                           unsigned int bufferTimesteps)
    {
        auto proxytype = idg::api::Type::CPU_REFERENCE;
        if (type == 0) {
            proxytype = idg::api::Type::CPU_REFERENCE;
        } else if (type == 1) {
            proxytype = idg::api::Type::CPU_OPTIMIZED;
        }
        return new idg::api::DegridderBufferImpl(proxytype, bufferTimesteps);
    }

    void DegridderBuffer_destroy(idg::api::DegridderBuffer* p) {
       delete p;
    }

    // void DegridderBuffer_request_visibilities(
    //     idg::DegridderBuffer* p,
    //     int timeIndex,
    //     int antenna1,
    //     int antenna2,
    //     double* uvwInMeters)
    // {
    //     p->request_visibilities(
    //         timeIndex,
    //         antenna1,
    //         antenna2,
    //         uvwInMeters);
    // }

    int DegridderBuffer_request_visibilities_with_rowid(
        idg::api::DegridderBuffer* p,
        int rowId,
        int timeIndex,
        int antenna1,
        int antenna2,
        double* uvwInMeters)
    {
        bool data_avail = p->request_visibilities(
            rowId,
            timeIndex,
            antenna1,
            antenna2,
            uvwInMeters);
        if (data_avail) return 1;
        else return 0;
    }

//     void DegridderBuffer_read_visibilities(
//         idg::DegridderBuffer* p,
//         int timeIndex,
//         int antenna1,
//         int antenna2,
//         void* visibilities) // ptr to complex<float>
//     {
//         p->read_visibilities(
//             timeIndex,
//             antenna1,
//             antenna2,
//             (complex<float>*) visibilities);
//     }


//     void DegridderBuffer_transform_grid(
//         idg::api::DegridderBuffer* p,
//         double crop_tolarance,
//         int nr_polarizations,
//         int height,
//         int width,
//         void *grid)
//     {
//         p->transform_grid(
//             crop_tolarance,
//             nr_polarizations,
//             height,
//             width,
//             (complex<double> *) grid);
//     }

} // extern C
