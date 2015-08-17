/** 
 *  \class CPU
 *
 *  \brief Class for ... 
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_H_
#define IDG_CPU_H_

#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "AbstractProxy.h"
#include "Kernels.h"

// Parameters for routines
#define SUBGRIDDER_PARAMETERS       unsigned nr_subgrids, float w_offset, void *uvw, void *wavenumbers, \
                                    void *visibilities, void *spheroidal, void *aterm, \
                                    void *metadata, void *subgrids
#define GRIDDER_PARAMETERS          float w_offset, void *uvw, void *wavenumbers, \
                                    void *visibilities, void *spheroidal, void *aterm, \
                                    void *grid
#define ADDER_PARAMETERS            unsigned nr_subgrids, void *metadata, void *subgrids, void *grid
#define SPLITTER_PARAMETERS         unsigned nr_subgrids, void *metadata, void *subgrids, void *grid
#define FFT_PARAMETERS              void *grid, int direction

namespace idg {
  
  namespace proxy {

    class CPU : public Proxy {
    public:
      
      /// Constructors
      CPU(Compiler compiler, 
	  Compilerflags flags,
	  Parameters params,
      ProxyInfo info = default_info()); 
      
      
      CPU(CompilerEnvironment cc, 
	  Parameters params,
	  ProxyInfo info = default_info()); 
      
      /// Copy constructor, copy assigment (see below in private)
      // te be edited
      
      /// Destructor
      ~CPU();
      
      // Get default values for ProxyInfo
      static ProxyInfo default_info();

      /** \brief Grid the visibilities onto a uniform grid (visibilities -> grid)
       *  \param w_offset [in] ... what is; format
       *  \param uvw [in] ... what is; format
       *  \param wavenumbers [in] ... what is; format
       *  \param visibilities [in] ... what is; format
       *  \param spheroidal [in] ... what is; format
       *  \param aterm [in] ... what is; format
       *  \param grid [out] ... what is; format
       */
      void grid_visibilities(
                 float w_offset,
			     void *uvw, 
                 void *wavenumbers,
                 void *visibilities, 
			     void *spheroidal, 
			     void *aterm, 
			     void *grid);

      /** \brief Degrid the visibilities from a uniform grid (grid -> visibilities)
       *  \param w_offset [in] ... what is; format
       *  \param uvw [in] ... what is; format
       *  \param wavenumbers [in] ... what is; format
       *  \param visibilities [in] ... what is; format
       *  \param spheroidal [in] ... what is; format
       *  \param aterm [in] ... what is; format
       *  \param metadata [in] ... what is; format
       *  \param grid [out] ... what is; format
       */
      void degrid_visibilities(
                 float w_offset,
			     void *uvw, 
                 void *wavenumbers,
                 void *visibilities, 
			     void *spheroidal, 
			     void *aterm, 
			     void *grid);
      
      /** \brief Applyies (inverse) Fourier transform to grid (grid -> grid)
       *  \param direction [in] idg::FourierDomainToImageDomain or idg::ImageDomainToFourierDomain
       *  \param grid [in/out] ...
       */
      void transform(DomainAtoDomainB direction, void* grid);


      // get parameters
      const Parameters& get_parameters() const { return mParams; }  
      const ProxyInfo& get_info() const { return mInfo; } 


    protected:
      // the function are divided into the following subroutines
      // gridder 
      void grid_onto_subgrids(int jobsize, SUBGRIDDER_PARAMETERS);

      void add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS);

      void split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS); 

      void degrid_from_subgrids(int jobsize, SUBGRIDDER_PARAMETERS);

      void run_gridder(int jobsize, SUBGRIDDER_PARAMETERS);

      void run_adder(int jobsize, ADDER_PARAMETERS);

      void run_splitter(int jobsize, SPLITTER_PARAMETERS);

      void run_degridder(int jobsize, SUBGRIDDER_PARAMETERS);

      void run_fft(FFT_PARAMETERS);

    private:
      
      void compile(Compiler compiler, Compilerflags flags);
      void parameter_sanity_check();
      void load_shared_objects();
      void find_kernel_functions();

      // data
      Parameters mParams;  // store parameters passed on creation
      ProxyInfo mInfo; // info about shared object files

      // store the ptr to Module, which each loads an .so-file 
      // std::vector< std::unique_ptr<runtime::Module> > modules;  
      std::vector<runtime::Module*> modules;  
      std::map<std::string,int> which_module;  
    }; // class CPU

  } // namespace proxy

} // namespace idg

#endif
