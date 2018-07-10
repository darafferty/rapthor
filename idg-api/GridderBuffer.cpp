/*
 * GridderBuffer.h
 * Access to IDG's high level gridder routines
 */


#include "GridderBufferImpl.h"
#include "BufferSetImpl.h"

#include <mutex>

using namespace std;

namespace idg {
namespace api {

    GridderBufferImpl::GridderBufferImpl(
        BufferSetImpl* bufferset,
        proxy::Proxy* proxy,
        size_t bufferTimesteps)
        : BufferImpl(bufferset, proxy, bufferTimesteps),
          m_bufferUVW2(0,0),
          m_bufferStationPairs2(0),
          m_buffer_weights(0,0,0,0),
          m_buffer_weights2(0,0,0,0),
          m_default_aterm_correction(bufferset->m_default_aterm_correction),
          m_avg_aterm_correction(bufferset->m_avg_aterm_correction),
          m_average_beam(bufferset->m_average_beam),
          m_do_gridding(bufferset->m_do_gridding),
          m_do_compute_avg_beam(bufferset->m_do_compute_avg_beam),
          m_apply_aterm(bufferset->m_apply_aterm)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        m_aterm_offsets2 = m_default_aterm_offsets;
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
        complex<float>* visibilities,
        const float* weights)
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
        std::copy(weights, weights + m_nr_channels*4, &m_buffer_weights(local_bl, local_time, 0, 0));
    }

    void GridderBufferImpl::flush_thread_worker()
    {
        if (m_do_compute_avg_beam)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < (m_subgridsize * m_subgridsize); i++)
            {
                for (int n = 0; n < m_aterm_offsets2.size() - 1; n++)
                {
                    int time_start = m_aterm_offsets2[n];
                    int time_end = m_aterm_offsets2[n+1];

                    // loop over baselines
                    for (size_t bl = 0; bl < m_nr_baselines; bl++)
                    {
                        unsigned int antenna1 = m_bufferStationPairs2(bl).first;
                        unsigned int antenna2 = m_bufferStationPairs2(bl).second;

                        float sum_of_weights[4] = {};
                        for(size_t t=time_start; t < time_end; t++)
                        {
                            if (std::isinf(m_bufferUVW2(bl,t).u)) continue;

                            for(int ch=0; ch < get_frequencies_size(); ch++)
                            {
                                for(int pol = 0; pol < 4; pol++)
                                {
                                    sum_of_weights[pol] += m_buffer_weights2(bl, t, ch, pol);
                                }
                            }
                        }

                        // add kronecker product to average beam

                        size_t offset = m_subgridsize * m_subgridsize * m_nrStations * n;
                        size_t offset1 = offset + antenna1 * m_subgridsize * m_subgridsize;
                        size_t offset2 = offset + antenna2 * m_subgridsize * m_subgridsize;

                        std::complex<float> kp[16] = {};
                        kp[0] = conj(m_aterms2[offset2 + i].xx)*m_aterms2[offset1 + i].xx;
                        kp[1] = conj(m_aterms2[offset2 + i].xx)*m_aterms2[offset1 + i].xy;
                        kp[2] = conj(m_aterms2[offset2 + i].xy)*m_aterms2[offset1 + i].xx;
                        kp[3] = conj(m_aterms2[offset2 + i].xy)*m_aterms2[offset1 + i].xy;

                        kp[4] = conj(m_aterms2[offset2 + i].xx)*m_aterms2[offset1 + i].yx;
                        kp[5] = conj(m_aterms2[offset2 + i].xx)*m_aterms2[offset1 + i].yy;
                        kp[6] = conj(m_aterms2[offset2 + i].xy)*m_aterms2[offset1 + i].yx;
                        kp[7] = conj(m_aterms2[offset2 + i].xy)*m_aterms2[offset1 + i].yy;

                        kp[ 8] = conj(m_aterms2[offset2 + i].yx)*m_aterms2[offset1 + i].xx;
                        kp[ 9] = conj(m_aterms2[offset2 + i].yx)*m_aterms2[offset1 + i].xy;
                        kp[10] = conj(m_aterms2[offset2 + i].yy)*m_aterms2[offset1 + i].xx;
                        kp[11] = conj(m_aterms2[offset2 + i].yy)*m_aterms2[offset1 + i].xy;

                        kp[12] = conj(m_aterms2[offset2 + i].yx)*m_aterms2[offset1 + i].yx;
                        kp[13] = conj(m_aterms2[offset2 + i].yx)*m_aterms2[offset1 + i].yy;
                        kp[14] = conj(m_aterms2[offset2 + i].yy)*m_aterms2[offset1 + i].yx;
                        kp[15] = conj(m_aterms2[offset2 + i].yy)*m_aterms2[offset1 + i].yy;

                        for(size_t ii = 0; ii < 4; ii++)
                        {
                            for(size_t jj = 0; jj < 4; jj++)
                            {
                                m_average_beam[i*16 + ii*4 + jj] +=
                                    sum_of_weights[0] * conj(kp[ii  ]) * kp[jj  ] +
                                    sum_of_weights[1] * conj(kp[ii+4]) * kp[jj+4] +
                                    sum_of_weights[2] * conj(kp[ii+8]) * kp[jj+8] +
                                    sum_of_weights[3] * conj(kp[ii+12]) * kp[jj+12];
                            }
                        }
                    }
                }
            }
        }

        if (!m_do_gridding) return;

        Array4D<std::complex<float>> *aterm_correction;
        if (m_apply_aterm)
        {
            aterm_correction = &m_avg_aterm_correction;
        }
        else
        {
            m_aterm_offsets_array = Array1D<unsigned int>(m_default_aterm_offsets.data(), m_default_aterm_offsets.size());
            m_aterms_array = Array4D<Matrix2x2<complex<float>>>(m_default_aterms.data(), m_default_aterm_offsets.size()-1, m_nrStations, m_subgridsize, m_subgridsize);
            aterm_correction = &m_default_aterm_correction;
        }

        Plan::Options options;

        options.w_step = m_wStepInLambda;
        options.nr_w_layers = m_nr_w_layers;
        options.plan_strict = false;

        const int nr_channel_groups = m_channel_groups.size();
        Plan* plans[nr_channel_groups];
        std::mutex locks[nr_channel_groups];
        for (int i = 0; i < nr_channel_groups; i++) {
            locks[i].lock();
        }
        omp_set_nested(true);

        /*
         * Start two threads:
         *  thread 0: create plans
         *  thread 1: execute these plans
         */
        #pragma omp parallel num_threads(2)
        {
            // Create plans
            if (omp_get_thread_num() == 0) {
                for (int i = 0; i < nr_channel_groups; i++) {
                    std::cout << "planning channels: " << m_channel_groups[i].first << "-" << m_channel_groups[i].second << std::endl;
                    Plan* plan = new Plan(
                        m_kernel_size,
                        m_subgridsize,
                        m_gridHeight,
                        m_cellHeight,
                        m_grouped_frequencies[i],
                        m_bufferUVW2,
                        m_bufferStationPairs2,
                        m_aterm_offsets_array,
                        options);


                    plans[i] = plan;
                    locks[i].unlock();
                } // end for i
            } // end create plans

            // Execute plans
            if (omp_get_thread_num() == 1) {
                for (int i = 0; i < nr_channel_groups; i++) {
                    // Wait for plan to become available
                    locks[i].lock();
                    Plan *plan = plans[i];

                    if (i == 0) {
                        Array3D<Visibility<std::complex<float>>>& visibilities_src = m_bufferVisibilities2[i];

                        m_proxy->initialize(
                            *plan,
                            m_wStepInLambda,
                            m_cellHeight,
                            m_kernel_size,
                            m_subgridsize,
                            m_grouped_frequencies[i],
                            visibilities_src,
                            m_bufferUVW2,
                            m_bufferStationPairs2,
                            *m_grid,
                            m_aterms_array,
                            m_aterm_offsets_array,
                            m_spheroidal);
                    }

                    // Start flush
                    std::cout << "gridding channels: " << m_channel_groups[i].first << "-" << m_channel_groups[i].second << std::endl;
                    Array3D<Visibility<std::complex<float>>>& visibilities_src = m_bufferVisibilities2[i];
                    auto nr_baselines = visibilities_src.get_z_dim();
                    auto nr_timesteps = visibilities_src.get_y_dim();
                    auto nr_channels  = visibilities_src.get_x_dim();
                    Array3D<Visibility<std::complex<float>>> visibilities_dst(
                            m_visibilities.data(), nr_baselines, nr_timesteps, nr_channels);
                    memcpy(
                        visibilities_dst.data(),
                        visibilities_src.data(),
                        visibilities_src.bytes());

                    m_proxy->run_gridding(
                        *plan,
                        m_wStepInLambda,
                        m_cellHeight,
                        m_kernel_size,
                        m_subgridsize,
                        m_grouped_frequencies[i],
                        m_bufferVisibilities2[i],
                        m_bufferUVW2,
                        m_bufferStationPairs2,
                        *m_grid,
                        m_aterms_array,
                        m_aterm_offsets_array,
                        m_spheroidal);
                } // end for i
                // Wait for all plans to be executed
                m_proxy->finish_gridding();
            } // end execute plans

        } // end omp parallel

        // Cleanup plans
        for (int i = 0; i < nr_channel_groups; i++) {
            delete plans[i];
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
        std::swap(m_buffer_weights, m_buffer_weights2);
        std::swap(m_aterm_offsets, m_aterm_offsets2);
        m_aterm_offsets_array = Array1D<unsigned int>(m_aterm_offsets2.data(), m_aterm_offsets2.size());

        std::swap(m_aterms, m_aterms2);
        assert(m_aterms2.size()==(m_aterm_offsets_array.get_x_dim()-1)*m_nrStations*m_subgridsize*m_subgridsize);
        m_aterms_array = Array4D<Matrix2x2<complex<float>>>(m_aterms2.data(), m_aterm_offsets_array.get_x_dim()-1, m_nrStations, m_subgridsize, m_subgridsize);

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
    void GridderBufferImpl::reset_aterm()
    {
      if (m_aterm_offsets.size()!=2) {
        m_aterm_offsets = std::vector<unsigned int>(2, 0);
      }
      m_aterm_offsets[0] = 0;
      m_aterm_offsets[1] = m_bufferTimesteps;

      size_t n_old_aterms = m_aterm_offsets2.size()-1; // Nr aterms in previous chunk

      size_t atermBlockSize = m_nrStations*m_subgridsize*m_subgridsize;
      m_aterms.resize(atermBlockSize);
      std::copy(m_aterms2.data()+(n_old_aterms-1)*atermBlockSize,
                m_aterms2.data()+(n_old_aterms)*atermBlockSize,
                (Matrix2x2<complex<float>>*) m_aterms.data());
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
// //         resize2f(static_cast<int>(m_subgridsize),
// //                  static_cast<int>(m_subgridsize),
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
        
        m_bufferUVW2 = Array2D<UVWCoordinate<float>>(m_nr_baselines, m_bufferTimesteps);
        m_bufferVisibilities2.clear();
        for (auto & channel_group : m_channel_groups)
        {
            int nr_channels = channel_group.second - channel_group.first;
            m_bufferVisibilities2.push_back(Array3D<Visibility<std::complex<float>>>(m_nr_baselines, m_bufferTimesteps, nr_channels));
        }
        m_bufferStationPairs2 = Array1D<std::pair<unsigned int,unsigned int>>(m_nr_baselines);
        m_buffer_weights = Array4D<float>(m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4);
        m_buffer_weights2 = Array4D<float>(m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4);
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
