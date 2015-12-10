#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()

#include "idg-config.h"
#include "SandyBridgeEP.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

        /// Constructors
        SandyBridgeEP::SandyBridgeEP(
            Parameters params,
            Compiler compiler,
            Compilerflags flags,
            ProxyInfo info)
            : CPU(params, compiler, flags, info)
        {
            #if defined(DEBUG)
            cout << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
        }

        ProxyInfo SandyBridgeEP::default_info()
        {
            #if defined(DEBUG)
            cout << __func__ << endl;
            #endif

            string  srcdir = string(IDG_INSTALL_DIR)
                + "/lib/kernels/CPU/SandyBridgeEP";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif
 
            // Create temp directory
            string tmpdir = CPU::make_tempdir();
 
            // Create proxy info
            ProxyInfo p = CPU::default_proxyinfo(srcdir, tmpdir);

            return p;
        }

        
        string SandyBridgeEP::default_compiler() 
        {
            #if defined(USING_GNU_CXX_COMPILER)
            return "g++";
            #else 
            return "icpc";
            #endif
        }
        

        string SandyBridgeEP::default_compiler_flags() 
        {
            string debug = "Debug";
            string relwithdebinfo = "RelWithDebInfo";

            #if defined(USING_GNU_CXX_COMPILER)
            if (debug == IDG_BUILD_TYPE)
                return "-Wall -g -fopenmp -lfftw3f";
            else if (relwithdebinfo == IDG_BUILD_TYPE)
                return "-O3 -g -fopenmp -lfftw3f";
            else
                return "-Wall -O3 -fopenmp -lfftw3f";
            #else 
            // Settings (general, assuming intel as default) 
            if (debug == IDG_BUILD_TYPE)
                return "-Wall -openmp -g -DDEBUG -mkl -lmkl_avx -lmkl_vml_avx -mavx";
            else if (relwithdebinfo == IDG_BUILD_TYPE)
                return "-O3 -g -openmp -mkl -lmkl_avx -lmkl_vml_avx -mavx";
            else
                return "-Wall -O3 -openmp -mkl -lmkl_avx -lmkl_vml_avx -mavx";
            #endif
        }

        } // namespace cpu
    } // namespace proxy
} // namespace idg





// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::cpu::SandyBridgeEP CPU_SandyBridgeEP;

    CPU_SandyBridgeEP* CPU_SandyBridgeEP_init(
                unsigned int nr_stations,
                unsigned int nr_channels,
                unsigned int nr_timesteps,
                unsigned int nr_timeslots,
                float        imagesize,
                unsigned int grid_size,
                unsigned int subgrid_size)
    {
        idg::Parameters P;
        P.set_nr_stations(nr_stations);
        P.set_nr_channels(nr_channels);
        P.set_nr_timesteps(nr_timesteps);
        P.set_nr_timeslots(nr_timeslots);
        P.set_imagesize(imagesize);
        P.set_subgrid_size(subgrid_size);
        P.set_grid_size(grid_size);

        return new CPU_SandyBridgeEP(P);
    }

    void CPU_SandyBridgeEP_grid(CPU_SandyBridgeEP* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            void *aterm,
                            void *spheroidal)
    {
         p->grid_visibilities(
                (const std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) metadata,
                (std::complex<float>*) grid,
                w_offset,
                (const std::complex<float>*) aterm,
                (const float*) spheroidal);
    }

    void CPU_SandyBridgeEP_degrid(CPU_SandyBridgeEP* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            void *aterm,
                            void *spheroidal)
    {
         p->degrid_visibilities(
                (std::complex<float>*) visibilities,
                    (const float*) uvw,
                    (const float*) wavenumbers,
                    (const int*) metadata,
                    (const std::complex<float>*) grid,
                    w_offset,
                    (const std::complex<float>*) aterm,
                    (const float*) spheroidal);
     }

    void CPU_SandyBridgeEP_transform(CPU_SandyBridgeEP* p,
                    int direction,
                    void *grid)
    {
       if (direction!=0)
           p->transform(idg::ImageDomainToFourierDomain,
                    (std::complex<float>*) grid);
       else
           p->transform(idg::FourierDomainToImageDomain,
                    (std::complex<float>*) grid);
    }

    void CPU_SandyBridgeEP_destroy(CPU_SandyBridgeEP* p) {
       delete p;
    }

}  // end extern "C"
