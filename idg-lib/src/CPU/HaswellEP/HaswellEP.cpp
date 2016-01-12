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
#include "HaswellEP.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

        /// Constructors
        HaswellEP::HaswellEP(
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

        ProxyInfo HaswellEP::default_info()
        {
            #if defined(DEBUG)
            cout << __func__ << endl;
            #endif

            string  srcdir = string(IDG_INSTALL_DIR)
                + "/lib/kernels/CPU/HaswellEP";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif

            // Create temp directory
            string tmpdir = CPU::make_tempdir();

            // Create proxy info
            ProxyInfo p = CPU::default_proxyinfo(srcdir, tmpdir);

            return p;
        }


        string HaswellEP::default_compiler()
        {
            #if defined(USING_GNU_CXX_COMPILER)
            return "g++";
            #else
            return "icpc";
            #endif
        }


        string HaswellEP::default_compiler_flags()
        {
            string debug = "Debug";
            string relwithdebinfo = "RelWithDebInfo";

            #if defined(USING_GNU_CXX_COMPILER)
            // Settings for gcc
            if (debug == IDG_BUILD_TYPE)
                return "-Wall -g -fopenmp -lfftw3f";
            else if (relwithdebinfo == IDG_BUILD_TYPE)
                return "-O3 -g -fopenmp -lfftw3f";
            else
                return "-Wall -O3 -fopenmp -lfftw3f";
            #else
            // Settings (general, assuming intel as default)
            // TODO: investigate effect of -march=core-avx2 option (performance, accuracy)
            if (debug == IDG_BUILD_TYPE)
                return "-Wall -g -openmp -xcore-avx2 -mkl";
            else if (relwithdebinfo == IDG_BUILD_TYPE)
                return "-O3 -g -openmp -xcore-avx2 -mkl";
            else
                return "-Wall -O3 -openmp -xcore-avx2 -mkl";
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
    typedef idg::proxy::cpu::HaswellEP CPU_HaswellEP;

    CPU_HaswellEP* CPU_HaswellEP_init(
                unsigned int nr_stations,
                unsigned int nr_channels,
                unsigned int nr_time,
                unsigned int nr_timeslots,
                float        imagesize,
                unsigned int grid_size,
                unsigned int subgrid_size)
    {
        idg::Parameters P;
        P.set_nr_stations(nr_stations);
        P.set_nr_channels(nr_channels);
        P.set_nr_time(nr_time);
        P.set_nr_timeslots(nr_timeslots);
        P.set_imagesize(imagesize);
        P.set_subgrid_size(subgrid_size);
        P.set_grid_size(grid_size);

        return new CPU_HaswellEP(P);
    }

    void CPU_HaswellEP_grid(CPU_HaswellEP* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *baselines,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->grid_visibilities(
                (const std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) baselines,
                (std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void CPU_HaswellEP_degrid(CPU_HaswellEP* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *baselines,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->degrid_visibilities(
                (std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) baselines,
                (const std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void CPU_HaswellEP_transform(CPU_HaswellEP* p,
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

    void CPU_HaswellEP_destroy(CPU_HaswellEP* p) {
       delete p;
    }

}  // end extern "C"
