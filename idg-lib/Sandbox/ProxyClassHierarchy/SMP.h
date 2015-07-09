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

      void add_subgrids_to_grid(int jobsize, void *uvw, void *subgrids, void *grid);

      void split_grid_into_subgrids(int jobsize, void *uvw, void *subgrids, void *grid); 

      void degrid_from_subgrids(int jobsize, void *wavenumbers, void *aterm, 
				void *baselines, void *visibilities, void *uvw, 
				void *spheroidal, void *subgrids); 

    protected:

      void run_gridder(int jobsize, void *visibilities, void *uvw, 
		       void *wavenumbers, void *aterm, void *spheroidal, 
		       void *baselines, void *subgrids);

      void run_adder(int jobsize, void *uvw, void *subgrids, void *grid);

      void run_splitter(int jobsize, void *uvw, void *subgrids, void *grid);

      void run_degridder(int jobsize, void *wavenumbers, void *aterm, 
			 void *baselines, void *visibilities, void *uvw, 
			 void *spheroidal, void *subgrids);

    private:
      
      void compile(Compiler compiler, Compilerflags flags);

      // data
      Parameters mParams;  // store parameters passed on creation
      ProxyInfo mInfo; // info about shared object files

    }; // class SMP

  } // namespace proxy

} // namespace idg

#endif



