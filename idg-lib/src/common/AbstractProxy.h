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

#define GRIDDER_PARAMETERS   unsigned nr_subgrids, float w_offset, \
                             void *uvw, void *wavenumbers, \
                             void *visibilities, void *spheroidal, void *aterm, \
                             void *metadata, void *subgrids
#define GRIDDER_ARGUMENTS    nr_subgrids, w_offset, uvw, wavenumbers, visibilities, \
                             spheroidal, aterm, metadata, subgrids
#define DEGRIDDER_PARAMETERS GRIDDER_PARAMETERS
#define ADDER_PARAMETERS     unsigned nr_subgrids, void *metadata, void *subgrids, void *grid
#define SPLITTER_PARAMETERS  ADDER_PARAMETERS
#define FFT_PARAMETERS       void *grid, int direction

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

        /** Contructors of derived classes should support
         * Proxy(Compiler compiler,
         *       Compilerflags flags,
         *       Parameters params);
         *       ProxyInfo info = default_info());
         *
         * Proxy(CompilerEnviroment cc,
         *       Parameters params);
         *       ProxyInfo info = default_info());
         */

        /// Destructor
        virtual ~Proxy() {}; // default for now; can make purely virtual?

        /// Inteface for methods provided by the proxy
        /** \brief Grid the visibilities onto uniform subgrids (visibilities -> subgrids) */
        void grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS);

        /** \brief Add subgrids to a gridd (subgrids -> grid) */
        void add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS);

        /** \brief Exctract subgrids from a grid (grid -> subgrids) */
        void split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS);

        /** \brief Degrid the visibilities from uniform subgrids (subgrids -> visibilities) */
        void degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS);

        /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
           *  \param direction [in] idg::FourierDomainToImageDomain or idg::ImageDomainToFourierDomain
           *  \param grid [in/out] ...
        */
        void transform(DomainAtoDomainB direction, void* grid);
    };

  } // namespace proxy

} // namespace idg

#endif
