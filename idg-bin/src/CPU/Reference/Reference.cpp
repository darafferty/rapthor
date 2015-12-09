#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "Reference.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            Reference::Reference(
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


            ProxyInfo Reference::default_info()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                string  srcdir = string(IDG_INSTALL_DIR)
                    + "/lib/kernels/CPU/Reference";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = CPU::make_tempdir();

                // Create proxy info
                ProxyInfo p = CPU::default_proxyinfo(srcdir, tmpdir);

                return p;
            }


            string Reference::default_compiler()
            {
                #if defined(USING_INTEL_CXX_COMPILER)
                return "icpc";
                #else
                return "g++";
                #endif
            }


            string Reference::default_compiler_flags()
            {
                string debug = "Debug";
                string relwithdebinfo = "RelWithDebInfo";

                #if defined(USING_INTEL_CXX_COMPILER)
                // Settings for the intel compiler
                if (debug == IDG_BUILD_TYPE)
                    return "-Wall -g -DDEBUG -openmp -mkl -lmkl_def";
                else if (relwithdebinfo == IDG_BUILD_TYPE)
                    return "-O3 -openmp -g -mkl -lmkl_def";
                else
                    return "-Wall -O3 -openmp -mkl -lmkl_def";
                #else
                // Settings (general, assuming gcc as default)
                if (debug == IDG_BUILD_TYPE)
                    return "-Wall -g -DDEBUG -fopenmp -lfftw3f";
                else if (relwithdebinfo == IDG_BUILD_TYPE)
                    return "-O3 -g -fopenmp -lfftw3f";
                else
                    return "-Wall -O3 -fopenmp -lfftw3f";
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
    typedef idg::proxy::cpu::Reference CPU_Reference;

    CPU_Reference* CPU_Reference_init(
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

        return new CPU_Reference(P);
    }

    void CPU_Reference_grid(CPU_Reference* p,
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

    void CPU_Reference_degrid(CPU_Reference* p,
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

    void CPU_Reference_transform(CPU_Reference* p,
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

    void CPU_Reference_destroy(CPU_Reference* p) {
       delete p;
    }

}  // end extern "C"
