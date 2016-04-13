/**
 *  \class Proxy
 *
 *  \brief Abstract base class for all "proxy clases"
 *
 *  All classes inherited from Proxy will adhere to a common
 *  inteface. All instances P have the common functionality of:
 *  P.grid_visibilities(arguments)
 *  P.degrid_visibilities(arguments)
 *  P.transform(arguments)
 *  as well as a set of getter and setter routines for parameters.
 */

#ifndef IDG_PROXY_H_
#define IDG_PROXY_H_

#include <complex>
#include <vector>
#include <limits>
#include <cstring>

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"  // to be use in derived class
#include "Parameters.h" // to be use in derived class
#include "Types.h"
#include "Plan.h"

namespace idg {
    enum DomainAtoDomainB {
        FourierDomainToImageDomain,
        ImageDomainToFourierDomain
    };

    /// typedefs
    typedef std::string Compiler;
    typedef std::string Compilerflags;
}


namespace idg {
    namespace proxy {

        class Proxy
        {
        public:
            /*
                High level routines
            */
            /** \brief Grid the visibilities onto a uniform grid
             * Using:
             * ST = NR_STATIONS
             * BL = NR_BASELINES
             * CH = NR_CHANNELS
             * TS = NR_TIMESLOTS
             * TI = NR_TIMESTEPS*NR_TIMESLOTS
             * PL = NR_POLARIZATIONS
             * GS = GRIDSIZE
             * SB = SUBGRIDSIZE
             * \param visibilities [in] complex<float>[BL][TI][CH][PL]
             * \param uvw [in] float[3*BL][TI]
             * \param wavenumbers [in] float[CH]
             * \param baselines [in] int[BL][2]
             * \param grid [out] complex<float>[PL][GS][GS]
             * \param w_offset [in] float
             * \param kernel_size [in] int
             * \param aterm_offsets [in] int[TS]
             * \param aterm [in] complex<float>[ST][TI][PL][SB][SB]
             * \param spheroidal [in] float[SB][SB]
             */
            virtual void grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal) = 0;

            /** \brief Degrid the visibilities from a uniform grid
             *  (grid -> visibilities)
             * Using:
             * ST = NR_STATIONS
             * BL = NR_BASELINES
             * CH = NR_CHANNELS
             * TS = NR_TIMESLOTS
             * TI = NR_TIMESTEPS*NR_TIMESLOTS
             * PL = NR_POLARIZATIONS
             * GS = GRIDSIZE
             * SB = SUBGRIDSIZE
             * \param visibilities [out] complex<float>[BL][TI][CH][PL]
             * \param uvw [in] float[3*BL][TI]
             * \param wavenumbers [in] float[CH]
             * \param baselines [in] int[BL][2]
             * \param grid [in] complex<float>[PL][GS][GS]
             * \param w_offset [in] float
             * \param kernel_size [in] int
             * \param aterm [in] complex<float>[ST][TI][PL][SB][SB]
             * \param aterm_offsets [in] int[TS]
             * \param spheroidal [in] float[SB][SB]
             */
            virtual void degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal) = 0;

            /** \brief Applyies (inverse) Fourier transform to grid
             *  (grid -> grid)
             *  \param direction [in] idg::FourierDomainToImageDomain
             *                     or idg::ImageDomainToFourierDomain
             *  \param grid [in/out] complex<float>[PL][GS][GS]
             */
            virtual void transform(
                DomainAtoDomainB direction,
                std::complex<float>* grid) = 0;

            // Auxiliary: set and get methods
            unsigned int get_nr_stations() const {
                return mParams.get_nr_stations(); }
            unsigned int get_nr_baselines() const {
                return mParams.get_nr_baselines(); }
            unsigned int get_nr_channels() const {
                return mParams.get_nr_channels(); }
            unsigned int get_nr_time() const {
                return mParams.get_nr_time(); }
            unsigned int get_nr_timeslots() const {
                return mParams.get_nr_timeslots(); }
            float get_imagesize() const {
                return mParams.get_imagesize(); }
            unsigned int get_grid_size() const {
                return mParams.get_grid_size(); }
            unsigned int get_subgrid_size() const {
                return mParams.get_subgrid_size(); }
            unsigned int get_job_size() const {
                return mParams.get_job_size(); }
            unsigned int get_job_size_gridding() const {
                return mParams.get_job_size_gridding(); }
            unsigned int get_job_size_degridding() const {
                return mParams.get_job_size_degridding(); }
            unsigned int get_job_size_gridder() const {
                return mParams.get_job_size_gridder(); }
            unsigned int get_job_size_adder() const {
                return mParams.get_job_size_adder(); }
            unsigned int get_job_size_splitter() const {
                return mParams.get_job_size_splitter(); }
            unsigned int get_job_size_degridder() const {
                return mParams.get_job_size_degridder(); }
            unsigned int get_nr_polarizations() const {
                return mParams.get_nr_polarizations(); }

            void set_job_size(unsigned int js) {
                mParams.set_job_size(js); }
            void set_job_size_gridding(unsigned int js) {
                mParams.set_job_size_gridding(js); }
            void set_job_size_degridding(unsigned int js) {
                mParams.set_job_size_degridding(js); }

        public:
            // creates and execution plan given the data and
            // the proxy's parameters
            Plan create_plan(
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const int *aterm_offsets,
                const int kernel_size,
                const int max_nr_timesteps = std::numeric_limits<int>::max());

        protected:
            Parameters mParams;  // store parameters passed on creation
        };

    } // namespace proxy
} // namespace idg

#endif
