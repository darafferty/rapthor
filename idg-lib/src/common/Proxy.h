/**
 *  \class Proxy
 *
 *  \brief Abstract base class for all "proxy clases"
 *
 *  Have a more detailed description here
 */

#ifndef IDG_PROXY_H_
#define IDG_PROXY_H_

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"  // to be use in derived class
#include "Parameters.h" // to be use in derived class
#include "CompilerEnvironment.h" // to be use in derived class

// High level method parameters
#define GRIDDING_PARAMETERS void *uvw, \
                            void *wavenumbers, \
                            void *visibilities, \
                            void *spheroidal, \
                            void *aterm, \
                            void *metadata, \
                            void *grid
#define DEGRIDDING_PARAMETERS GRIDDING_PARAMETERS

// Low level method parameters
#define GRIDDER_PARAMETERS   unsigned nr_subgrids, float w_offset, \
                             void *uvw, void *wavenumbers, \
                             void *visibilities, void *spheroidal, void *aterm, \
                             void *metadata, void *subgrids
#define DEGRIDDER_PARAMETERS GRIDDER_PARAMETERS
#define ADDER_PARAMETERS     unsigned nr_subgrids, void *metadata, void *subgrids, void *grid
#define SPLITTER_PARAMETERS  ADDER_PARAMETERS
#define FFT_PARAMETERS       void *grid, int direction

// Low level method arguments
#define GRIDDER_ARGUMENTS    nr_subgrids, w_offset, uvw, wavenumbers, visibilities, \
                             spheroidal, aterm, metadata, subgrids
#define DEGRIDDER_ARGUMENT   GRIDDER_ARGUMENTS
#define ADDER_ARGUMENTS      nr_subgrids, metadata, subgrids, grid
#define SPLITTER_ARGUMENTS   ADDER_ARGUMENTS
#define FFT_ARGUMENTS        grid, direction

namespace idg {
    enum DomainAtoDomainB {
        FourierDomainToImageDomain,
        ImageDomainToFourierDomain
    };
}


namespace idg {

    namespace proxy {

    class Proxy {
        public:
            /*
                High level routines
                These routines operate on grids
            */
            /** \brief Grid the visibilities onto a uniform grid (visibilities -> grid)
             *  \param uvw [in] ... what is; format
             *  \param wavenumbers [in] ... what is; format
             *  \param visibilities [in] ... what is; format
             *  \param aterm [in] ... what is; format
             *  \param metadata [in] ... what is; format
             *  \param grid [out] ... what is; format
             */
            virtual void grid_visibilities(GRIDDING_PARAMETERS) = 0;

            /** \brief Degrid the visibilities from a uniform grid (grid -> visibilities)
             *  \param uvw [in] ... what is; format
             *  \param wavenumbers [in] ... what is; format
             *  \param visibilities [out] ... what is; format
             *  \param aterm [in] ... what is; format
             *  \param metadata [in] ... what is; format
             *  \param grid [in] ... what is; format
             */
            virtual void degrid_visibilities(DEGRIDDING_PARAMETERS) = 0;

            /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
               *  \param direction [in] idg::FourierDomainToImageDomain or idg::ImageDomainToFourierDomain
               *  \param grid [in/out] ...
            */
            virtual void transform(DomainAtoDomainB direction, void* grid) = 0;

            /*
                Low level routines
                These routines operate on subgrids
            */
            /// Inteface for methods provided by the proxy
            virtual void grid_onto_subgrids(GRIDDER_PARAMETERS) = 0;

            /** \brief Add subgrids to a gridd (subgrids -> grid) */
            virtual void add_subgrids_to_grid(ADDER_PARAMETERS) = 0;

            /** \brief Exctract subgrids from a grid (grid -> subgrids) */
            virtual void split_grid_into_subgrids(SPLITTER_PARAMETERS) = 0;

            /** \brief Degrid the visibilities from uniform subgrids (subgrids -> visibilities) */
            virtual void degrid_from_subgrids(DEGRIDDER_PARAMETERS) = 0;

        protected:
            Parameters mParams;  // store parameters passed on creation
    };

  } // namespace proxy

} // namespace idg

#endif
