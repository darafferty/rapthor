/** 
 *  \class Proxy
 *
 *  \brief Abstract base class for all "proxy clases" 
 *
 *  Have a more detailed description here
 */

#ifndef IDG_PROXY_H_
#define IDG_PROXY_H_

#include <vector>
#include "RuntimeWrapper.h"
#include "ProxyInfo.h"  // to be use in derived class
#include "Parameters.h" // to be use in derived class
#include "CompilerEnvironment.h" // to be use in derived class

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
       *       AlgorithmParameters algparams = default_algparams()); 
       *       ProxyInfo info = default_info()); 
       *
       * Proxy(CompilerEnviroment cc, 
       *       Parameters params);
       *       AlgorithmParameters algparams = default_algparams()); 
       *       ProxyInfo info = default_info()); 
       */

      /// Copy constructor, copy assigment (see below in private)
      // te be edited

      /// Destructor
      virtual ~Proxy() {}; // default for now; can make purely virtual?


      /// Inteface for methods provided by the proxy
      
      /** \brief Grid the visibilities onto a uniform grid (visibilities -> grid)
       *  \param visibilities [in] ... what is; format
       *  \param uvw [in] ... what is; format
       *  \param wavenumbers [in] ... what is; format
       *  \param aterm [in] ... what is; format
       *  \param spheroidal [in] ... what is; format
       *  \param baselines [in] ... what is; format
       *  \param grid [out] ... what is; format
       */
      virtual void grid_visibilities(void *visibilities, 
				     void *uvw, 
				     void *wavenumbers,
				     void *aterm, 
				     void *spheroidal, 
				     void *baselines, 
				     void *grid) = 0;
      /// removed jobsize: do we need one for each routine (gridder, adder, splitter, degridder)
      /// instead of one for all

      /** \brief Degrid the visibilities from a uniform grid (grid -> visibilities)
       *  \param grid [in] ... what is; format
       *  \param uvw [in] ... what is; format
       *  \param wavenumbers [in] ... what is; format
       *  \param aterm [in] ... what is; format
       *  \param spheroidal [in] ... what is; format
       *  \param baselines [in] ... what is; format
       *  \param visibilities [out] ... what is; format
       */
      virtual void degrid_visibilities(void *grid,
				       void *uvw,
				       void *wavenumbers, 
				       void *aterm,
				       void *spheroidal, 
				       void *baselines,
				       void *visibilities) = 0;

      /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
       *  \param direction [in] idg::FourierDomainToImageDomain or idg::ImageDomainToFourierDomain
       *  \param grid [in/out] ...
       */
      virtual void transform(DomainAtoDomainB direction, void* grid) = 0;
            
    protected:  
      // runtime::Module *module;
      // Proxy(const Proxy&); // prevents copy for now; do somehow differently?
      // Proxy& operator=(const Proxy&);
    };

    
  } // namespace proxy

} // namespace idg

#endif
