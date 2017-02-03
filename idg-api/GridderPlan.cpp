/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */


#include "GridderPlan.h"


using namespace std;

namespace idg {

    GridderPlan::GridderPlan(Type architecture,
                             size_t bufferTimesteps)
        : Scheme(architecture, bufferTimesteps),
          m_bufferUVW2(0,0),
          m_bufferStationPairs2(0),
          m_bufferVisibilities2(0,0,0)
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
        if (m_flush_thread.joinable()) m_flush_thread.join();
    }


    void GridderPlan::grid_visibilities(
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

        copy(visibilities, visibilities + get_frequencies_size() * m_nrPolarizations,
             (complex<float>*) &m_bufferVisibilities(local_bl, local_time, 0));
    }

    void flush_thread_worker( 
        proxy::Proxy *proxy,
        complex<float> *visibilities,
        float *uvw,
        const float *wavenumbers,
        int *station_pairs,
        std::complex<float> *grid,
        std::complex<double> *grid_double,
        const float w_offset,
        const int kernel_size,
        const std::complex<float> *aterm,
        const int *aterm_offsets,
        const float *spheroidal,
        int nr_polarizations,
        int grid_height,
        int grid_width
)
    
    {

        // TODO 
//                 void gridding(
//                     const float w_offset,
//                     const float cell_size,
//                     const unsigned int kernel_size,
//                     const Array1D<float>& frequencies,
//                     const Array3D<Visibility<std::complex<float>>>& visibilities,
//                     const Array2D<UVWCoordinate<float>>& uvw,
//                     const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
//                     Array3D<std::complex<float>>& grid,
//                     const Array4D<Matrix2x2<std::complex<float>>>& aterms,
//                     const Array1D<unsigned int>& aterms_offsets,
//                     const Array2D<float>& spheroidal);
//         
//         
//         proxy->grid_visibilities(
//             visibilities,
//             uvw,
//             wavenumbers,
//             station_pairs,
//             grid,
//             w_offset,
//             7, //kernel_size,
//             aterm,
//             aterm_offsets,
//             spheroidal);
// 
        // TODO: remove below //////////////////////////
        // HACK: Add results to double precision grid
        for (auto p = 0; p < nr_polarizations; ++p) {
            for (auto y = 0; y < grid_height; ++y) {
                for (auto x = 0; x < grid_width; ++x) {
                    grid_double[p*grid_height*grid_width + y*grid_width + x] += grid[p*grid_height*grid_width + y*grid_width + x];
                    grid[p*grid_height*grid_width + y*grid_width + x] = 0;
                }
            }
        }
    }
    
    // Must be called whenever the buffer is full or no more data added
    void GridderPlan::flush()
    {
        // Return if no input in buffer
        if (m_timeindices.size() == 0) return;


        // if there is still a flushthread running, wait for it to finish
        if (m_flush_thread.joinable()) m_flush_thread.join();
        
        // int kernSize = max(m_wKernelSize, m_aKernelSize, m_spheroidalKernelSize);
        int kernelSize = m_wKernelSize;

        
        m_bufferUVW2 = std::move(m_bufferUVW);
        m_bufferStationPairs2 = std::move(m_bufferStationPairs); 
        m_bufferVisibilities2 = std::move(m_bufferVisibilities);

//         TODO
//         m_flush_thread = std::thread (
//             flush_thread_worker,
//             m_proxy,
//             (complex<float>*) m_bufferVisibilities2.data(),
//             (float*)m_bufferUVW2.data(),
//             (float*) m_wavenumbers.data(),
//             (int*) m_bufferStationPairs2.data(),
//             m_grid.data(),
//             m_grid_double,
//             m_wOffsetInLambda,
//             kernelSize,
//             (complex<float>*) m_aterms.data(),
//             m_aterm_offsets,
//             m_spheroidal.data(),
//             m_nrPolarizations,
//             m_gridHeight,
//             m_gridWidth);
        
        // The primary buffers are now empty
        // they need to be allocated again
        
        m_bufferUVW = Array2D<UVWCoordinate<float>>(m_nrGroups, m_bufferTimesteps);
        m_bufferVisibilities = Array3D<Visibility<std::complex<float>>>(m_nrGroups, m_bufferTimesteps, get_frequencies_size());
        m_bufferStationPairs = Array1D<std::pair<unsigned int,unsigned int>>(m_nrGroups);

        // Prepare next batch
        m_timeStartThisBatch += m_bufferTimesteps;
        m_timeStartNextBatch += m_bufferTimesteps;
        m_timeindices.clear();
        set_uvw_to_infinity();
    }
    
    void GridderPlan::finished()
    {
        flush();
        // if there is still a flushthread running, wait for it to finish
        if (m_flush_thread.joinable()) m_flush_thread.join();
    }


    void GridderPlan::transform_grid(
        double crop_tolerance,
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<double> *grid)
    {
        // Normal case: no arguments -> transform member grid
        // Note: the other case is to perform the transform on a copy
        // so that the process can be monitored
        if (grid == nullptr) {
            nr_polarizations = m_nrPolarizations;
            height           = m_gridHeight;
            width            = m_gridWidth;
            grid             = m_grid_double;
        }

        // Inverse FFT complex-to-complex for each polarization
        ifft_grid(nr_polarizations, height, width, grid);

        // Apply the spheroidal and scale
        // assuming only half the visibilities are gridded:
        // mulitply real part by 2, set imaginary part to zero,
        // Note: float, because m_spheroidal is in float to match the
        // lower level API
        Grid2D<float> spheroidal_grid(height, width);
        resize2f(static_cast<int>(m_subgridSize),
                 static_cast<int>(m_subgridSize),
                 m_spheroidal.data(),
                 static_cast<int>(height),
                 static_cast<int>(width),
                 spheroidal_grid.data());

        const double c_real = 2.0 / (height * width);
        for (auto pol = 0; pol < nr_polarizations; ++pol) {
            for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                    double scale = c_real/spheroidal_grid(y,x);
                    if (spheroidal_grid(y,x) < crop_tolerance) {
                        scale = 0.0;
                    }
                    grid[pol*height*width + y*width + x] =
                        grid[pol*height*width + y*width + x].real() * scale;
                }
            }
        }
    }


} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    idg::GridderPlan* GridderPlan_init(
        unsigned int type,
        unsigned int bufferTimesteps)
    {
        auto proxytype = idg::Type::CPU_REFERENCE;
        if (type == 0) {
            proxytype = idg::Type::CPU_REFERENCE;
        } else if (type == 1) {
            proxytype = idg::Type::CPU_OPTIMIZED;
        }
        return new idg::GridderPlan(proxytype, bufferTimesteps);
    }

    void GridderPlan_destroy(idg::GridderPlan* p) {
       delete p;
    }

    void GridderPlan_grid_visibilities(
        idg::GridderPlan* p,
        int     timeIndex,
        int     antenna1,
        int     antenna2,
        double* uvwInMeters,
        float*  visibilities) // size CH x PL x 2
    {
        p->grid_visibilities(
            timeIndex,
            antenna1,
            antenna2,
            uvwInMeters,
            (complex<float>*) visibilities); // size CH x PL
    }

    void GridderPlan_transform_grid(
        idg::GridderPlan* p,
        double crop_tol,
        int    nr_polarizations,
        int    height,
        int    width,
        void*  grid)
    {
        p->transform_grid(
            crop_tol,
            nr_polarizations,
            height,
            width,
            (complex<double> *) grid);
    }

} // extern C
