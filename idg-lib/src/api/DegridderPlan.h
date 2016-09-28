/**
 * DegridderPlan.h
 *
 * \class DegridderPlan
 *
 * \brief Access to IDG's high level gridder routines
 *
 * The DegridderPlan manages two buffers: a request buffer and a visibility buffer.
 * The former holds UVW data and antenna IDs for the visibilities to predict later.
 * Once filled with requests, the visibilities can be predicted, which fills the
 * visibility buffer. Now one can read the visibility buffer, and signal that the
 * visibility buffer can be overwritten again with another compute operation.
 *
 * Usage (pseudocode):
 *
 * idg::DegridderPlan plan(...);
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

#ifndef IDG_DEGRIDDERPLAN_H_
#define IDG_DEGRIDDERPLAN_H_

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
#include "Scheme.h"

namespace idg {

    class DegridderPlan : public Scheme
    {
    public:
        // Constructors and destructor
        DegridderPlan(Type architecture = Type::CPU_REFERENCE,
                      size_t bufferTimesteps = 4096);

        virtual ~DegridderPlan();

        /** \brief Request a visibility to compute later
         *  \param rowId [in] unique identifier used to read the data
         *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
         *                        or 0 <= timeIndex < bufferTimesteps
         *  \param antenna1 [in]  0 <= antenna1 < nrStations
         *  \param antenna2 [in]  antenna1 < antenna2 < nrStations
         *  \param uvwInMeters [in] (u, v, w)
         *  \return buffer_full [out] true, if request buffer is already full
         */
        bool request_visibilities(
            size_t rowId,
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            const double* uvwInMeters);

        /** \brief Request a visibility without using the rowId identifier
         *  Note: if this method is used for requesting, reading must be done
         *  without the rowId identifier as well
         */
        bool request_visibilities(
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
        std::vector<size_t> compute();

        /** \brief Signal that the visibilities can be overwritten */
        void finished_reading();

        /** \brief Read the visibilities from the visibility buffer
         *  \param rowId [in] identifier used in the request
         *  \param visibilities [out] complex<float>[NR_CHANNELS][NR_POLARIZATIONS]
         */
        void read_visibilities(size_t rowId, std::complex<float>* visibilities);

        /** \brief Read the visibilities from the visibility buffer without rowId
         *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
         *                        or 0 <= timeIndex < bufferTimesteps
         *  \param antenna1 [in]  0 <= antenna1 < nrStations
         *  \param antenna2 [in]  antenna1 < antenna2 < nrStations
         *  \param visibilities [out] complex<float>[NR_CHANNELS][NR_POLARIZATIONS]
         */
        void read_visibilities(
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            std::complex<float>* visibilities);

        /** \brief Transform the grid; normal use without arguments
         * Paremeters are need as transform is done on an external grid
         * i.e. on a copy
         * param crop_tolerance [in] ...
         * param nr_polarizations [in] number of correlations (normally 4)
         * param height [in] width in pixel
         * param width [in] width in pixel
         * param grid [in] complex<double>[nr_polarizations][height][width]
         */
        virtual void transform_grid(
            double crop_tolerance      = 5e-3,
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<double> *grid = nullptr) override;

        /** \brief Alias to call transform_grid()
         * param crop_tolerance [in] ...
         * param nr_polarizations [in] number of correlations (normally 4)
         * param height [in] width in pixel
         * param width [in] width in pixel
         * param grid [in] complex<double>[nr_polarizations][height][width]
         */
        virtual void image_to_fourier(
            double crop_tolerance      = 5e-3,
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<double> *grid = nullptr)
        {
            transform_grid(crop_tolerance, nr_polarizations,
                           height, width, grid);
        }

        /** \brief Explicitly flush the buffer */
        virtual void flush() override;

        // Set and get methods

        /** \brief Set the image to be used for predicting visibilities
         *  \param [in] nr_polarizations number of correlations (normally 4)
         *  \param [in] size size in pixels of a square image
         *  \param [in] grid std::complex<double>[nr_polarizations][size][size]
         */
        void set_image(
            size_t nr_polarizations,
            size_t size,
            std::complex<double>* grid)
        {
            set_grid(nr_polarizations, size, grid);
        }

        /** \brief Set the image to be used for predicting visibilities
         *  \param [in] nr_polarizations number of correlations (normally 4)
         *  \param [in] height hight in pixels
         *  \param [in] width width in pixels
         *  \param [in] grid std::complex<double>[nr_polarizations][size][size]
         */
        void set_image(
            size_t nr_polarizations,
            size_t height,
            size_t width,
            std::complex<double>* grid)
        {
            set_grid(nr_polarizations, height, width, grid);
        }

        size_t get_image_height() const { return get_grid_height(); };
        size_t get_image_width() const { return get_grid_width(); } ;

        bool is_request_buffer_full() const { return m_buffer_full; }
        bool is_data_marked_as_read() const { return m_data_read; }

    private:

        // Data
        bool m_buffer_full;
        bool m_data_read;
        std::vector<size_t> m_row_ids_to_read;
        std::map<size_t,std::pair<size_t,int>> m_row_ids_to_indices;
    };

} // namespace idg

#endif
