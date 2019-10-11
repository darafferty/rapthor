/**
 * GridderBuffer.h
 *
 * \class GridderBuffer
 *
 * \brief Access to IDG's high level gridder routines
 *
 * The GridderBuffer manages a buffer of a fixed number of time steps
 * One fills the buffer, which fill occasionally be flushed to grid
 * the visibilities onto the grid.
 *
 * Usage (pseudocode):
 *
 * idg::GridderBuffer plan(...);
 * plan.set_grid(grid);
 * plan.set_other_properties(...);
 * plan.bake();
 *
 * for (auto row = 0; row < nr_rows; ++row) {
 *    gridder.grid_visibilities(...);
 * }
 *
 * // Make sure no visibilites are still in the buffer
 * gridder.finished();
 *
 * // Transform the gridded visibilities to an image
 * gridder.transform_grid();
 *
 */

#ifndef IDG_GRIDDERBUFFERIMPL_H_
#define IDG_GRIDDERBUFFERIMPL_H_

#include <complex>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <thread>

#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif

#include "GridderBuffer.h"
#include "BufferImpl.h"

namespace idg {
namespace api {

    class BufferSetImpl;

    class GridderBufferImpl : public virtual GridderBuffer, public BufferImpl
    {
    public:
        // Constructors and destructor
        GridderBufferImpl(
            BufferSetImpl *bufferset,
            proxy::Proxy* proxy,
            size_t bufferTimesteps);

        virtual ~GridderBufferImpl();

        /** \brief Adds the visibilities to the buffer
         *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
         *                        or 0 <= timeIndex < bufferTimesteps
         *  \param antenna1 [in]  0 <= antenna1 < nrStations
         *  \param antenna2 [in]  antenna1 < antenna2 < nrStations
         *  \param uvwInMeters [in] double[3]: (u, v, w)
         *  \param visibilities [in] std::complex<float>[NR_CHANNELS][NR_POLARIZATIONS]
         */
        void grid_visibilities(
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            const double* uvwInMeters,
            std::complex<float>* visibilities,
            const float* weights);

        /** \brief Computes average beam
         */
        void compute_avg_beam();

        /** \brief Signal that not more visibilies are gridded */
        virtual void finished() override;

        /** \brief Explicitly flush the buffer */
        virtual void flush() override;

        /** \brief Transform the grid; normal use without arguments
         * No arguments => perform on grid set by set_grid()
         * Paremeters are need as transform is done on an external grid
         * i.e. on a copy
         * param crop_tolerance [in] ...
         * param nr_polarizations [in] number of correlations (normally 4)
         * param height [in] width in pixel
         * param width [in] width in pixel
         * param grid [in] complex<double>[nr_polarizations][height][width]
         */

        /** reset_aterm() Resets the new aterm for the next time chunk */
        virtual void reset_aterm();

    protected:
        virtual void malloc_buffers();

    private:

        //secondary buffers      
        Array2D<UVW<float>> m_bufferUVW2;                       // BL x TI
        Array1D<std::pair<unsigned int,unsigned int>> m_bufferStationPairs2;                         // BL
        Array3D<Visibility<std::complex<float>>> m_bufferVisibilities2;   // BL x TI x CH
        std::vector<Matrix2x2<std::complex<float>>> m_aterms2; // ST x SB x SB
        Array4D<float> m_buffer_weights;   // BL x TI x NR_CHANNELS x NR_POLARIZATIONS
        Array4D<float> m_buffer_weights2;   // BL x TI x NR_CHANNELS x NR_POLARIZATIONS
        std::vector<unsigned int>  m_aterm_offsets2;


        std::thread m_flush_thread;
        void flush_thread_worker();

        // references to members of parent BufferSet
        std::vector<std::complex<float>> &m_average_beam;
        Array4D<std::complex<float>> &m_default_aterm_correction;
        Array4D<std::complex<float>> &m_avg_aterm_correction;
        bool &m_do_gridding;
        bool &m_do_compute_avg_beam;
        bool &m_apply_aterm;
    };

} // namespace api
} // namespace idg

#endif
