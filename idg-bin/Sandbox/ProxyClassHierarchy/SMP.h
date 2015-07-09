
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
	  Parameters params,
          ProxyInfo info = default_info()); 
      
      
      SMP(CompilerEnvironment cc, 
	  Parameters params,
	  ProxyInfo info = default_info()); 
      
      /// Copy constructor, copy assigment (see below in private)
      // te be edited
      
      /// Destructor
      ~SMP();
      
      // Get default values for ProxyInfo
      static ProxyInfo default_info();

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


      // the function are divided into the following subroutines
      // gridder 
      void grid_onto_subgrids(int jobsize, void *visibilities, void *uvw, 
			      void *wavenumbers, void *aterm, void *spheroidal, 
			      void *baselines, void *subgrids);

      // adder
      // splitter
      // degridder

    protected:

      void run_gridder(int jobsize, void *visibilities, void *uvw, 
		       void *wavenumbers, void *aterm, void *spheroidal, 
		       void *baselines, void *subgrids);
      // virtual void add_subgrids_to_grid(); // adder
      // virtual void split_grid_into_subgrids(); // splitter
      // virtual void degrid_from_subgrids(); // degridder


    private:
      
      void compile(Compiler compiler, Compilerflags flags);

      // data
      Parameters mParams;
      ProxyInfo mInfo;

    }; // class SMP

  } // namespace proxy

} // namespace idg

#endif



