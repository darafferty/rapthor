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

#include <dlfcn.h>
#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "AbstractProxy.h"
#include "Kernels.h"

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
    
                /// Destructor
                ~CPU();
    
                // Get default values 
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
    
                // Get parameters of proxy
                const Parameters& get_parameters() const { return mParams; }
                const ProxyInfo& get_info() const { return mInfo; }
    
    
            // High level routines
            public:
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
        
            // Low level routines
            protected:
                virtual void run_gridder(int jobsize, GRIDDER_PARAMETERS);
        
                virtual void run_adder(int jobsize, ADDER_PARAMETERS);
        
                virtual void run_splitter(int jobsize, SPLITTER_PARAMETERS);
        
                virtual void run_degridder(int jobsize, DEGRIDDER_PARAMETERS);
        
                virtual void run_fft(FFT_PARAMETERS);

            protected:
                static std::string make_tempdir();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

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
