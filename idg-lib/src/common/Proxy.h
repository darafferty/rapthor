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
#define GRIDDER_PARAMETERS   unsigned nr_subgrids, float w_offset, \
                             void *uvw, void *wavenumbers, \
                             void *visibilities, void *spheroidal, void *aterm, \
                             void *metadata, void *subgrids
#define DEGRIDDER_PARAMETERS GRIDDER_PARAMETERS
#define ADDER_PARAMETERS     unsigned nr_subgrids, void *metadata, void *subgrids, void *grid
#define SPLITTER_PARAMETERS  ADDER_PARAMETERS
#define FFT_PARAMETERS       void *grid, int direction

// High level method arguments
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
            /// Inteface for methods provided by the proxy
            /** \brief Grid the visibilities onto uniform subgrids 
                (visibilities -> subgrids) 
                TODO: specify parameters, what is input, output, 
                input and output, and what does it mean
                \param wavenumbers [in]
                \param visibilities [in] 
                \param spheroidal [in]
                \param aterm [in]
                \param metadata [in]
                \param subgrids [out]
            */
            virtual void grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS) = 0;
    
            /** \brief Add subgrids to a gridd (subgrids -> grid) */
            virtual void add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS) = 0;
    
            /** \brief Exctract subgrids from a grid (grid -> subgrids) */
            virtual void split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS) = 0;
    
            /** \brief Degrid the visibilities from uniform subgrids (subgrids -> visibilities) */
            virtual void degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS) = 0;
    
            /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
               *  \param direction [in] idg::FourierDomainToImageDomain or idg::ImageDomainToFourierDomain
               *  \param grid [in/out] ...
            */
            virtual void transform(DomainAtoDomainB direction, void* grid) = 0;

        protected:
            Parameters mParams;  // store parameters passed on creation
    };

  } // namespace proxy

} // namespace idg

#endif
