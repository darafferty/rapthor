/**
 * DegridderBuffer.h
 *
 * \class DegridderBuffer
 *
 * \brief Access to IDG's high level gridder routines
 *
 * The DegridderBuffer manages two buffers: a request buffer and a visibility buffer.
 * The former holds UVW data and antenna IDs for the visibilities to predict later.
 * Once filled with requests, the visibilities can be predicted, which fills the
 * visibility buffer. Now one can read the visibility buffer, and signal that the
 * visibility buffer can be overwritten again with another compute operation.
 *
 * Usage (pseudocode):
 *
 * idg::DegridderBuffer plan(...);
 * plan.set_image(image);
 * plan.set_other_properties(...);
 * plan.bake();
 *
 * // Tranform the image to the Fourier domain
 * degridder.image_to_fourier();
 *
 * for (auto row = 0; row < nr_rows; ++row) {
 *
 *     bool is_buffer_full = degridder.request_visibilities(...);
 *
 *     // Request buffer is full, compute visibilities and read them
 *     // from the visibility buffer
 *     if (is_buffer_full || row == nr_rows-1) {
 *
 *          // Compute the requested visibilities
 *          auto available_row_ids = degridder.compute();
 *
 *          // Read all available visibilities
 *          for (auto& r : available_row_ids) {
 *              degridder.read_visibilities(r, visibilities);
 *              do_something(visibilities);
 *          }
 *
 *          // Signal that we can start requesting again
 *          degridder.finished_reading();
 *
 *          // Push failed request again (always fits) before continue the loop
 *          degridder.request_visibilities(...);
 *     }
 * } // for each row
 *
 */

#ifndef IDG_DEGRIDDERBUFFERIMPL_H_
#define IDG_DEGRIDDERBUFFERIMPL_H_

#include <complex>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include <stdexcept>
#include <cmath>

#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif

#include "DegridderBuffer.h"
#include "BufferImpl.h"

namespace idg {
namespace api {

    class DegridderBufferImpl : public virtual DegridderBuffer, public BufferImpl
    {
    public:
        // Constructors and destructor
        DegridderBufferImpl(
            BufferSetImpl *bufferset,
            proxy::Proxy* proxy,
            size_t bufferTimesteps = 4096);

        virtual ~DegridderBufferImpl();

        /** \brief Request a visibility to compute later
         *  \param rowId [in] unique identifier used to read the data
         *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
         *                        or 0 <= timeIndex < bufferTimesteps
         *  \param antenna1 [in]  0 <= antenna1 < nrStations
         *  \param antenna2 [in]  antenna1 < antenna2 < nrStations
         *  \param uvwInMeters [in] (u, v, w)
         *  \return buffer_full [out] true, if request buffer is already full
         */
        virtual bool request_visibilities(
            size_t rowId,
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            const double* uvwInMeters);

        /** \brief Request a visibility without using the rowId identifier
         *  Note: if this method is used for requesting, reading must be done
         *  without the rowId identifier as well
         */
        virtual bool request_visibilities(
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            const double* uvwInMeters)
        {
            return request_visibilities(0, timeIndex, antenna1,
                                 antenna2, uvwInMeters);
        }

        /** \brief Compute visibility of the requested visibilities
         *  \return list_of_rowIds [out] a list of all computed rowIds
         */
        virtual std::vector<std::pair<size_t, std::complex<float>*>> compute();

        /** \brief Signal that the visibilities can be overwritten */
        virtual void finished_reading();

        /** \brief Explicitly flush the buffer */
        virtual void flush() override;

        size_t get_image_height() const { return get_grid_height(); };
        size_t get_image_width() const { return get_grid_width(); } ;

        bool is_request_buffer_full() const { return m_buffer_full; }
        bool is_data_marked_as_read() const { return m_data_read; }

        /** reset_aterm() Resets the new aterm for the next time chunk */
        virtual void reset_aterm();

    protected:
        virtual void malloc_buffers();

    private:

        // Data
        Array3D<Visibility<std::complex<float>>> m_bufferVisibilities2;   // BL x TI x CH
        bool m_buffer_full;
        bool m_data_read;
        std::vector<std::pair<size_t, std::complex<float>*>> m_row_ids_to_data;
    };

} // namespace api
} // namespace idg

#endif
