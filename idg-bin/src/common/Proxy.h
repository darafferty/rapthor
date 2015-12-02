/**
 *  \class Proxy
 *
 *  \brief Abstract base class for all "proxy clases"
 *
 *  Have a more detailed description here
 */

#ifndef IDG_PROXY_H_
#define IDG_PROXY_H_

#include <complex>

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"  // to be use in derived class
#include "Parameters.h" // to be use in derived class
#include "CompilerEnvironment.h" // to be use in derived class


// Low level method parameters
// TODO: remove these defines, or at least move them somewhere else
/* #define GRIDDER_PARAMETERS   unsigned nr_subgrids, float w_offset, \ */
/*                              void *uvw, void *wavenumbers, \ */
/*                              void *visibilities, void *spheroidal, void *aterm, \ */
/*                              void *metadata, void *subgrids */
/* #define DEGRIDDER_PARAMETERS GRIDDER_PARAMETERS */
/* #define ADDER_PARAMETERS     unsigned nr_subgrids, void *metadata, void *subgrids, void *grid */
/* #define SPLITTER_PARAMETERS  ADDER_PARAMETERS */
// #define FFT_PARAMETERS       void *grid, int direction

// Low level method arguments
/* #define GRIDDER_ARGUMENTS    nr_subgrids, w_offset, uvw, wavenumbers, visibilities, \ */
/*                              spheroidal, aterm, metadata, subgrids */
/* #define DEGRIDDER_ARGUMENT   GRIDDER_ARGUMENTS */
/* #define ADDER_ARGUMENTS      nr_subgrids, metadata, subgrids, grid */
/* #define SPLITTER_ARGUMENTS   ADDER_ARGUMENTS */
// #define FFT_ARGUMENTS        grid, direction


namespace idg {
    enum DomainAtoDomainB {
        FourierDomainToImageDomain,
        ImageDomainToFourierDomain
    };
}


namespace idg {
    namespace proxy {

        class Proxy
        {
        public:
            /*
                High level routines
                These routines operate on grids
            */
            /** \brief Grid the visibilities onto a uniform grid
             * Using:
             * ST = NR_STATIONS
             * BL = NR_BASELINES = NR_STATIONS*(NR_STATIONS-1)/2
             * CH = NR_CHANNELS
             * TI = NR_TIMESTEPS*NR_TIMESLOTS
             * PL = NR_POLARIZATIONS
             * GS = GRIDSIZE
             * SB = SUBGRIDSIZE
             * \param visibilities [in] complex<float>[BL][TI][CH][PL]
             * \param uvw [in] float[3*BL][TI]
             * \param wavenumbers [in] float[CH]
             * \param metadata [in] ... what is; format
             * \param grid [out] complex<float>[PL][GS][GS]
             * \param w_offset [in] float
             * \param aterm [in] complex<float>[ST][TI][PL][SB][SB]
             * \param spheroidal [in] float[SB][SB]
             */
            virtual void grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *metadata,
                std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal) = 0;

            /** \brief Degrid the visibilities from a uniform grid (grid -> visibilities)
             * Using:
             * ST = NR_STATIONS
             * BL = NR_BASELINES = NR_STATIONS*(NR_STATIONS-1)/2
             * CH = NR_CHANNELS
             * TI = NR_TIMESTEPS*NR_TIMESLOTS
             * PL = NR_POLARIZATIONS
             * GS = GRIDSIZE
             * SB = SUBGRIDSIZE
             * \param visibilities [out] complex<float>[BL][TI][CH][PL]
             * \param uvw [in] float[3*BL][TI]
             * \param wavenumbers [in]
             * \param metadata [in] ... what is; format
             * \param grid [in] complex<float>[PL][GS][GS]
             * \param aterm [in] complex<float>[ST][TI][PL][SB][SB]
             * \param spheroidal [in] float[SB][SB]
             */
            virtual void degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *metadata,
                const std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal) = 0;

            /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
             *  \param direction [in] idg::FourierDomainToImageDomain
             *                     or idg::ImageDomainToFourierDomain
             *  \param grid [in/out] complex<float>[PL][GS][GS]
             */
            virtual void transform(DomainAtoDomainB direction,
                                   std::complex<float>* grid) = 0;

        protected:
            Parameters mParams;  // store parameters passed on creation
    };

  } // namespace proxy
} // namespace idg

#endif
