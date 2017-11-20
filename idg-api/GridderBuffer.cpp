/*
 * GridderBuffer.h
 * Access to IDG's high level gridder routines
 */


#include "GridderBufferImpl.h"

using namespace std;

namespace idg {
namespace api {

    GridderBufferImpl::GridderBufferImpl(
        proxy::Proxy* proxy,
        size_t bufferTimesteps)
        : BufferImpl(proxy, bufferTimesteps),
          m_bufferUVW2(0,0),
          m_bufferStationPairs2(0)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }


    GridderBufferImpl::~GridderBufferImpl()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        if (m_flush_thread.joinable()) m_flush_thread.join();
    }


    void GridderBufferImpl::grid_visibilities(
        size_t timeIndex,
        size_t antenna1,
        size_t antenna2,
        const double* uvwInMeters,
        const complex<float>* visibilities)
    {
        // exclude auto-correlations
        if (antenna1 == antenna2) return;

        int    local_time = timeIndex - m_timeStartThisBatch;
        size_t local_bl   = baseline_index(antenna1, antenna2);

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
        m_bufferUVW(local_bl, local_time) = {
            static_cast<float>(uvwInMeters[0]),
            static_cast<float>(uvwInMeters[1]),
            static_cast<float>(uvwInMeters[2])
        };

        m_bufferStationPairs(local_bl) = {
            static_cast<int>(antenna1),
            static_cast<int>(antenna2)
        };

        for (int i = 0; i<m_channel_groups.size(); i++)
        {
            std::copy(visibilities + m_channel_groups[i].first * m_nrPolarizations,
                      visibilities + m_channel_groups[i].second * m_nrPolarizations,
                      (complex<float>*) &m_bufferVisibilities[i](local_bl, local_time, 0));
        }
    }

    // Set the a-term that starts validity at timeIndex
    void GridderBufferImpl::set_aterm(
        size_t timeIndex,
        const complex<float>* aterms)
    {
        int n_ants = m_aterms.get_z_dim();
        int subgridsize = m_aterms.get_y_dim();
        int local_time = timeIndex - m_timeStartThisBatch;
        int n_old_aterms = m_aterms.get_w_dim();
        // Overwrite last a-term if new timeindex same as one but last element aterm_offsets
        if (local_time == m_aterm_offsets(m_aterm_offsets.get_x_dim()-2)) {
          std::cout<<"m_aterms.bytes()="<<m_aterms.bytes()<<std::endl;
          std::copy(aterms,
                    aterms + n_ants*subgridsize*subgridsize*4,
                    (complex<float>*) m_aterms.data(n_old_aterms-1));
        } else {
          assert(local_time > m_aterm_offsets(m_aterm_offsets.get_x_dim()-2));

          // insert new timeIndex before the last element in m_aterm_offsets
          assert(m_aterm_offsets.get_x_dim() == n_old_aterms+1);
          m_aterm_offsets.resize(n_old_aterms+2);
          m_aterm_offsets(n_old_aterms+2-1) = m_bufferTimesteps;
          m_aterm_offsets(n_old_aterms+2-2) = local_time;
          // push back new a-term
          m_aterms.resize(n_old_aterms+1, n_ants, subgridsize, subgridsize);
          std::copy(aterms,
                    aterms + n_ants*subgridsize*subgridsize*4,
                    (complex<float>*) m_aterms.data(n_old_aterms));
        }
    }

    // Reset the a-term for a new buffer; copy the last a-term from the
    // previous buffer;
    void GridderBufferImpl::reset_aterm()
    {
      if (m_aterms.get_w_dim()==1) {
        // Nothing to do, there was only one a-term, it remains valid
        return;
      } else {
        // Remember the last a-term as the new a-term for next chunk
        Array4D<Matrix2x2<std::complex<float>>> new_aterms(1, m_nrStations, m_subgridSize, m_subgridSize);
        std::copy(m_aterms.data(m_aterms.get_w_dim()-1),
                  m_aterms.data(m_aterms.get_w_dim()-1)+m_nrStations*m_subgridSize*m_subgridSize,
                  new_aterms.data());
        m_aterms = std::move(new_aterms);
        m_aterm_offsets = Array1D<unsigned int>(2);
        m_aterm_offsets(0) = 0;
        m_aterm_offsets(1) = m_bufferTimesteps;
      }
    }

    void GridderBufferImpl::flush_thread_worker()
    {
        Plan::Options options;

        options.w_step = m_wStepInLambda;
        options.nr_w_layers = m_nr_w_layers;
        options.plan_strict = true;

        for (int i = 0; i<m_channel_groups.size(); i++)
        {
            std::cout << "gridding channels: " << m_channel_groups[i].first << "-" << m_channel_groups[i].second << std::endl;
            Plan plan(
                m_kernel_size,
                m_subgridSize,
                m_gridHeight,
                m_cellHeight,
                m_grouped_frequencies[i],
                m_bufferUVW2,
                m_bufferStationPairs2,
                m_aterm_offsets,
                options);

            m_proxy->gridding(
                plan,
                m_wStepInLambda,
                m_cellHeight,
                m_kernel_size,
                m_grouped_frequencies[i],
                m_bufferVisibilities2[i],
                m_bufferUVW2,
                m_bufferStationPairs2,
                std::move(*m_grid),
                m_aterms,
                m_aterm_offsets,
                m_spheroidal);

        }
    }

    // Must be called whenever the buffer is full or no more data added
    void GridderBufferImpl::flush()
    {
        // Return if no input in buffer
        if (m_timeindices.size() == 0) return;


        // if there is still a flushthread running, wait for it to finish
        if (m_flush_thread.joinable()) m_flush_thread.join();

        std::swap(m_bufferUVW, m_bufferUVW2);
        std::swap(m_bufferStationPairs, m_bufferStationPairs2); 
        std::swap(m_bufferVisibilities, m_bufferVisibilities2);
        //std::swap(m_aterm_offsets, m_aterm_offsets2);
        //std::swap(m_aterms, m_aterm2);

        m_flush_thread = std::thread(&GridderBufferImpl::flush_thread_worker, this);

        // Temporary fix: wait for thread and swap buffers back
        // Needed because the proxy needs to be called each time with the same buffers
//         m_flush_thread.join();
//         std::swap(m_bufferUVW, m_bufferUVW2);
//         std::swap(m_bufferStationPairs, m_bufferStationPairs2); 
//         std::swap(m_bufferVisibilities, m_bufferVisibilities2);

        // Prepare next batch
        m_timeStartThisBatch += m_bufferTimesteps;
        m_timeStartNextBatch += m_bufferTimesteps;
        m_timeindices.clear();
        reset_aterm();
        set_uvw_to_infinity();
    }
    
    void GridderBufferImpl::finished()
    {
        flush();
        // if there is still a flushthread running, wait for it to finish
        if (m_flush_thread.joinable()) m_flush_thread.join();
//         // TODO: remove below //////////////////////////
//         // HACK: Add results to double precision grid
//         for (auto p = 0; p < m_nrPolarizations; ++p) {
//             #pragma omp parallel for
//             for (auto y = 0; y < m_gridHeight; ++y) {
//                 for (auto x = 0; x < m_gridWidth; ++x) {
//                     m_grid_double[p*m_gridHeight*m_gridWidth + y*m_gridWidth + x] += m_grid(0, p, y, x);
//                     m_grid(0, p, y, x) = 0;
//                 }
//             }
//         }
    }


//     void GridderBufferImpl::transform_grid(
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
//         // Inverse FFT complex-to-complex for each polarization
//         ifft_grid(nr_polarizations, height, width, grid);
// 
//         // Apply the spheroidal and scale
//         // assuming only half the visibilities are gridded:
//         // mulitply real part by 2, set imaginary part to zero,
//         // Note: float, because m_spheroidal is in float to match the
//         // lower level API
// //         Grid2D<float> spheroidal_grid(height, width);
// //         resize2f(static_cast<int>(m_subgridSize),
// //                  static_cast<int>(m_subgridSize),
// //                  m_spheroidal.data(),
// //                  static_cast<int>(height),
// //                  static_cast<int>(width),
// //                  spheroidal_grid.data());
// // 
//         const double c_real = 2.0 / (height * width);
//         for (auto pol = 0; pol < nr_polarizations; ++pol) {
//             for (auto y = 0; y < height; ++y) {
//                 for (auto x = 0; x < width; ++x) {
//                     double scale = c_real; // /spheroidal_grid(y,x);
// //                     if (spheroidal_grid(y,x) < crop_tolerance) {
// //                         scale = 0.0;
// //                     }
//                     grid[pol*height*width + y*width + x] =
//                         grid[pol*height*width + y*width + x].real() * scale;
//                 }
//             }
//         }
//     }
    
    void GridderBufferImpl::malloc_buffers()
    {
        BufferImpl::malloc_buffers();
        
        m_bufferUVW2 = Array2D<UVWCoordinate<float>>(m_nrGroups, m_bufferTimesteps);
        m_bufferVisibilities2.clear();
        for (auto & channel_group : m_channel_groups)
        {
            int nr_channels = channel_group.second - channel_group.first;
            m_bufferVisibilities2.push_back(Array3D<Visibility<std::complex<float>>>(m_nrGroups, m_bufferTimesteps, nr_channels));
        }
        m_bufferStationPairs2 = Array1D<std::pair<unsigned int,unsigned int>>(m_nrGroups);
    }



} // namespace api
} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
// extern "C" {
//
//     idg::api::GridderBuffer* GridderBuffer_init(
//         unsigned int type,
//         unsigned int bufferTimesteps)
//     {
//         auto proxytype = idg::api::Type::CPU_REFERENCE;
//         if (type == 0) {
//             proxytype = idg::api::Type::CPU_REFERENCE;
//         } else if (type == 1) {
//             proxytype = idg::api::Type::CPU_OPTIMIZED;
//         }
//         return new idg::api::GridderBufferImpl(proxytype, bufferTimesteps);
//     }
//
//     void GridderBuffer_destroy(idg::api::GridderBuffer* p) {
//        delete p;
//     }
//
//     void GridderBuffer_grid_visibilities(
//         idg::api::GridderBuffer* p,
//         int     timeIndex,
//         int     antenna1,
//         int     antenna2,
//         double* uvwInMeters,
//         float*  visibilities) // size CH x PL x 2
//     {
//         p->grid_visibilities(
//             timeIndex,
//             antenna1,
//             antenna2,
//             uvwInMeters,
//             (complex<float>*) visibilities); // size CH x PL
//     }

//     void GridderBuffer_transform_grid(
//         idg::api::GridderBuffer* p,
//         double crop_tol,
//         int    nr_polarizations,
//         int    height,
//         int    width,
//         void*  grid)
//     {
//         p->transform_grid(
//             crop_tol,
//             nr_polarizations,
//             height,
//             width,
//             (complex<double> *) grid);
//     }

// } // extern C
