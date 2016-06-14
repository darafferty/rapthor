/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */


#include "GridderPlan.h"


using namespace std;

namespace idg {

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


    void GridderPlan::grid_visibilities(
        const complex<float>* visibilities,
        const double* uvwInMeters,
        size_t antenna1,
        size_t antenna2,
        size_t timeIndex)
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


        if (antenna1 > antenna2) swap(antenna1, antenna2);
        m_bufferStationPairs[local_bl] = {
            static_cast<int>(antenna1),
            static_cast<int>(antenna2)
        };

        copy(visibilities, visibilities + get_frequencies_size() * m_nrPolarizations,
             (complex<float>*) &m_bufferVisibilities(local_bl, local_time, 0));
    }


    // Must be called whenever the buffer is full or no more data added
    void GridderPlan::flush()
    {
        // Return if no input in buffer
        if (m_timeindices.size() == 0) return;

        // int kernSize = max(m_wKernelSize, m_aKernelSize, m_spheroidalKernelSize);
        int kernelSize = m_wKernelSize;

        // TODO: this routine should be not much more than this call
        m_proxy->grid_visibilities(
            (complex<float>*) m_bufferVisibilities.data(),
            (float*) m_bufferUVW.data(),
            (float*) m_wavenumbers.data(),
            (int*) m_bufferStationPairs.data(),
            m_grid.data(),
            m_wOffsetInLambda,
            kernelSize,
            (complex<float>*) m_aterms.data(),
            m_aterm_offsets,
            m_spheroidal.data());

        // TODO: remove below //////////////////////////
        // HACK: Add results to double precision grid
        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    m_grid_double[p*m_gridHeight*m_gridWidth
                                  + y*m_gridWidth + x] += m_grid(p, y, x);
                }
            }
        }
        // TODO: remove above //////////////////////////

        // Prepare next batch
        m_timeStartThisBatch += m_bufferTimesteps;
        m_timeStartNextBatch += m_bufferTimesteps;
        m_timeindices.clear();
        reset_buffers(); // optimization: only call "set_uvw_to_infinity()" here
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
        const double c_imag = 0.0;
        for (auto pol = 0; pol < nr_polarizations; ++pol) {
            for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                    complex<double> scale;
                    if (spheroidal_grid(y,x) >= crop_tolerance) {
                        scale = complex<double>(c_real/spheroidal_grid(y,x), c_imag);
                    } else {
                        scale = 0.0;
                    }
                    grid[pol*height*width + y*width + x] *= scale;
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

    void GridderPlan_transform_grid(
        idg::GridderPlan* p,
        double crop_tol,
        int nr_polarizations,
        int height,
        int width,
        void* grid)
    {
        p->transform_grid(
            crop_tol,
            nr_polarizations,
            height,
            width,
            (complex<double> *) grid);
    }

} // extern C
