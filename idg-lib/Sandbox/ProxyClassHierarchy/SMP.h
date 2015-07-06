
/** 
 *  \class Proxy
 *
 *  \brief Abstract base class for all "proxy clases" 
 *
 *  Have a more detailed description here
 */

#ifndef IDG_SMP_H_
#define IDG_SMP_H_

#include "Proxy.h"

namespace idg {

  namespace proxy {

    class SMP : public Proxy {
    public:

      /// Constructors
      SMP(Compiler compiler, 
	  Compilerflags flags,
	  CompileTimeConstants constants,
          ProxyInfo info = ProxyInfo()); 

      
      SMP(CompilerEnvironment cc, 
	  CompileTimeConstants constants,
	  ProxyInfo info = ProxyInfo()); 
      
      /// Copy constructor, copy assigment (see below in private)
      // te be edited

      /// Destructor
      ~SMP();


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
				     void *grid);

      /** \brief Degrid the visibilities from a uniform grid (grid -> visibilities)
       *  \param grid [in] ... what is; format
       *  \param uvw [in] ... what is; format
       *  \param wavenumbers [in] ... what is; format
       *  \param aterm [in] ... what is; format
       *  \param spheroidal [in] ... what is; format
       *  \param baselines [in] ... what is; format
       *  \param visibilities [out] ... what is; format
       */
      void degrid_visibilities(void *grid,
			       void *uvw,
			       void *wavenumbers, 
			       void *aterm,
			       void *spheroidal, 
			       void *baselines,
			       void *visibilities);
      
      /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
       *  \param direction [in] idg::FourierDomainToImageDomain or idg::ImageDomainToFourierDomain
       *  \param grid [in/out] ...
       */
      void transform(DomainAtoDomainB direction, void* grid);

    }; // class SMP

  } // namespace proxy

} // namespace idg

#endif



