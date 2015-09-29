/**
 *  \class OpenCL
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_OPENCL_H_
#define IDG_OPENCL_H_

#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include "AbstractProxy.h"
#include "Kernels.h"

// High level method parameters
#define CL_GRIDDER_PARAMETERS   cl::Context &context, unsigned nr_subgrids, float w_offset, \
                                cl::Buffer &h_uvw, cl::Buffer &d_wavenumbers, \
                                cl::Buffer &h_visibilities, cl::Buffer &d_spheroidal, cl::Buffer &d_aterm, \
                                cl::Buffer &h_metadata, cl::Buffer &h_subgrids
#define CL_DEGRIDDER_PARAMETERS CL_GRIDDER_PARAMETERS
#define CL_ADDER_PARAMETERS     cl::Context &context, unsigned nr_subgrids, cl::Buffer &h_metadata, cl::Buffer &h_subgrids, cl::Buffer &h_grid
#define CL_SPLITTER_PARAMETERS  CL_ADDER_PARAMETERS
#define CL_FFT_PARAMETERS       cl::Context &context, cl::Buffer &h_grid, int direction

// High level method arguments
#define CL_GRIDDER_ARGUMENTS    context, nr_subgrids, w_offset, h_uvw, d_wavenumbers, h_visibilities, \
                                d_spheroidal, d_aterm, h_metadata, h_subgrids
#define CL_DEGRIDDER_ARGUMENTS  CL_GRIDDER_ARGUMENTS
#define CL_ADDER_ARGUMENTS      context, nr_subgrids, h_metadata, h_subgrids, h_grid
#define CL_SPLITTER_ARGUMENTS   CL_ADDER_ARGUMENTS
#define CL_FFT_ARGUMENTS        context, h_grid, direction


namespace idg {
    namespace proxy {

        class OpenCL {
            public:
                /// Constructors
                OpenCL(Parameters params,
                    unsigned deviceNumber = 0,
                    Compilerflags flags = default_compiler_flags(),
                    ProxyInfo info = default_info());
    
                ~OpenCL();
    
                // Get default values 
                static ProxyInfo default_info();
                static std::string default_compiler_flags();
    
                // Get parameters of proxy
                const Parameters& get_parameters() const { return mParams; }
                const ProxyInfo& get_info() const { return mInfo; }
    
    
            // High level routines
            public:
                /** \brief Grid the visibilities onto uniform subgrids 
                           (visibilities -> subgrids). */
                void grid_onto_subgrids(CL_GRIDDER_PARAMETERS);
        
                /** \brief Add subgrids to a gridd 
                           (subgrids -> grid). */
                void add_subgrids_to_grid(CL_ADDER_PARAMETERS);
        
                /** \brief Exctract subgrids from a grid 
                           (grid -> subgrids). */
                void split_grid_into_subgrids(CL_SPLITTER_PARAMETERS);
        
                /** \brief Degrid the visibilities from uniform subgrids 
                           (subgrids -> visibilities). */
                void degrid_from_subgrids(CL_DEGRIDDER_PARAMETERS);
        
                /** \brief Applyies (inverse) Fourier transform to grid 
                           (grid -> grid).
                 *  \param direction [in] idg::FourierDomainToImageDomain or 
                                          idg::ImageDomainToFourierDomain
                 *  \param grid [in/out] ...
                 */
                void transform(DomainAtoDomainB direction, cl::Context &context, cl::Buffer &h_grid);
        
            // Low level routines
            protected:
                virtual void run_gridder(CL_GRIDDER_PARAMETERS);
        
                virtual void run_adder(CL_ADDER_PARAMETERS);
        
                virtual void run_splitter(CL_SPLITTER_PARAMETERS);
        
                virtual void run_degridder(CL_DEGRIDDER_PARAMETERS);
        
                virtual void run_fft(CL_FFT_PARAMETERS);

            protected:
                static std::string make_tempdir();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

                void compile(Compilerflags flags);
                void parameter_sanity_check();
                void load_shared_objects();
                void find_kernel_functions();
        
                // data
                cl::Device device;
                Parameters mParams; // remove if inherited from Proxy
                ProxyInfo mInfo; // info about shared object files
        
                std::vector<cl::Program*> programs;
                std::map<std::string,int> which_program;
        }; // class OpenCL
    
    } // namespace proxy
} // namespace idg

#endif
