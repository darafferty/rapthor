/**
 *  \class CUDA
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include <cuda.h>
#include "CU.h"
#include "AbstractProxy.h"
#include "Kernels.h"

// High level method parameters
#define CU_GRIDDER_PARAMETERS   cu::Context &context, unsigned nr_subgrids, float w_offset, \
                                cu::HostMemory &h_uvw, cu::DeviceMemory &d_wavenumbers, \
                                cu::HostMemory &h_visibilities, cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_aterm, \
                                cu::HostMemory &h_metadata, cu::HostMemory &h_subgrids
#define CU_DEGRIDDER_PARAMETERS CU_GRIDDER_PARAMETERS
#define CU_ADDER_PARAMETERS     cu::Context &context, unsigned nr_subgrids, cu::HostMemory &h_metadata, cu::HostMemory &h_subgrids, cu::HostMemory &h_grid
#define CU_SPLITTER_PARAMETERS  CU_ADDER_PARAMETERS
#define CU_FFT_PARAMETERS       cu::Context &context, cu::HostMemory &h_grid, int direction

// High level method arguments
#define CU_GRIDDER_ARGUMENTS    context, nr_subgrids, w_offset, h_uvw, d_wavenumbers, h_visibilities, \
                                d_spheroidal, d_aterm, h_metadata, h_subgrids
#define CU_DEGRIDDER_ARGUMENTS  CU_GRIDDER_ARGUMENTS
#define CU_ADDER_ARGUMENTS      context, nr_subgrids, h_metadata, h_subgrids, h_grid
#define CU_SPLITTER_ARGUMENTS   CU_ADDER_ARGUMENTS
#define CU_FFT_ARGUMENTS        context, h_grid, direction


namespace idg {
    namespace proxy {

        class CUDA {
            public:
                /// Constructors
                CUDA(Parameters params,
                    unsigned deviceNumber = 0,
                    Compiler compiler = default_compiler(),
                    Compilerflags flags = default_compiler_flags(),
                    ProxyInfo info = default_info());
    
                ~CUDA();
    
                // Get default values 
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
    
                // Get parameters of proxy
                const Parameters& get_parameters() const { return mParams; }
                const ProxyInfo& get_info() const { return mInfo; }
    
    
            // High level routines
            public:
                /** \brief Grid the visibilities onto uniform subgrids 
                           (visibilities -> subgrids). */
                void grid_onto_subgrids(int jobsize, CU_GRIDDER_PARAMETERS);
        
                /** \brief Add subgrids to a gridd 
                           (subgrids -> grid). */
                void add_subgrids_to_grid(int jobsize, CU_ADDER_PARAMETERS);
        
                /** \brief Exctract subgrids from a grid 
                           (grid -> subgrids). */
                void split_grid_into_subgrids(int jobsize, CU_SPLITTER_PARAMETERS);
        
                /** \brief Degrid the visibilities from uniform subgrids 
                           (subgrids -> visibilities). */
                void degrid_from_subgrids(int jobsize, CU_DEGRIDDER_PARAMETERS);
        
                /** \brief Applyies (inverse) Fourier transform to grid 
                           (grid -> grid).
                 *  \param direction [in] idg::FourierDomainToImageDomain or 
                                          idg::ImageDomainToFourierDomain
                 *  \param grid [in/out] ...
                 */
                void transform(DomainAtoDomainB direction, cu::Context &context, cu::HostMemory &h_grid);
        
            // Low level routines
            protected:
                virtual void run_gridder(int jobsize, CU_GRIDDER_PARAMETERS);
        
                virtual void run_adder(int jobsize, CU_ADDER_PARAMETERS);
        
                virtual void run_splitter(int jobsize, CU_SPLITTER_PARAMETERS);
        
                virtual void run_degridder(int jobsize, CU_DEGRIDDER_PARAMETERS);
        
                virtual void run_fft(CU_FFT_PARAMETERS);

            protected:
                static std::string make_tempdir();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

                void compile(Compiler compiler, Compilerflags flags);
                void parameter_sanity_check();
                void load_shared_objects();
                void find_kernel_functions();
        
                // data
                cu::Device device;
                Parameters mParams; // remove if inherited from Proxy
                ProxyInfo mInfo; // info about shared object files
        
                // store the ptr to Module, which each loads an .ptx-file
                std::vector<cu::Module*> modules;
                std::map<std::string,int> which_module;
        }; // class CUDA
    
    } // namespace proxy
} // namespace idg

#endif
