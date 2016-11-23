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
#include "Optimized.h"
#include "arch.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

        /// Constructors
        Optimized::Optimized(
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

        ProxyInfo Optimized::default_info()
        {
            #if defined(DEBUG)
            cout << __func__ << endl;
            #endif

            string  srcdir = string(IDG_INSTALL_DIR)
                + "/lib/kernels/CPU/Optimized";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif

            // Create temp directory
            string tmpdir = CPU::make_tempdir();

            // Create proxy info
            ProxyInfo p = CPU::default_proxyinfo(srcdir, tmpdir);

            return p;
        }


        string Optimized::default_compiler()
        {
            #if defined(GNU_CXX_COMPILER)
            return "g++";
            #elif defined(CLANG_CXX_COMPILER)
            return "clang++";
            #else
            return "icpc";
            #endif
        }


        string Optimized::default_compiler_flags()
        {
            stringstream flags;

            // Add build type flags
            string debug = "Debug";
            string relwithdebinfo = "RelWithDebInfo";
            if (debug == IDG_BUILD_TYPE) {
                flags << "-Wall -g";
            } else if (relwithdebinfo == IDG_BUILD_TYPE) {
                flags << "-O3 -g";
            } else {
                flags << "-Wall -O3";
            }

            // Intel compiler
            stringstream intel_flags;
            intel_flags << " -qopenmp -axcore-avx2 -mkl=parallel";
            #if defined(BUILD_WITH_PYTHON)
            // HACK: to make code be corretly loaded with ctypes
            intel_flags << " -lmkl_avx2 -lmkl_vml_avx2 -lmkl_avx -lmkl_vml_avx";
            #endif

            // GNU compiler
            stringstream gnu_flags;
            bool avx2_supported = check_4th_gen_intel_core_features();
            printf("AVX2 support: %d\n", avx2_supported);
            gnu_flags << " -std=c++11 -fopenmp -ffast-math";
            if (avx2_supported) {
                gnu_flags << " -march=core-avx2";
            }

            // Clang compiler
            stringstream clang_flags;
            clang_flags << " -std=c++11 -fopenmp";

            // MKL
            stringstream mkl_flags;
            #if defined(HAVE_MKL)
            mkl_flags << " -L" << MKL_LIB_DIR;
            mkl_flags << " -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core";
            mkl_flags << " -lmkl_avx2 -lmkl_vml_avx2";
            mkl_flags << " -lmkl_avx -lmkl_vml_avx";
            #endif

            // Add compiler specific flags
            stringstream compiler_flags;
            #if defined(GNU_CXX_COMPILER)
                flags << gnu_flags.str();
                #if defined(HAVE_MKL)
                flags << mkl_flags.str();
                #else
                flags << " -I" << FFTW3_INCLUDE_DIR << " " << FFTW3F_LIB;
                #endif
            #elif defined(CLANG_CXX_COMPILER)
                flags << clang_flags.str();
                #if defined(HAVE_MKL)
                flags << mkl_flags.str();
                #else
                flags << " -lfftw3f";
                #endif
            #else
                flags << intel_flags.str();
            #endif

            return flags.str();
        }

        } // namespace cpu
    } // namespace proxy
} // namespace idg





// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::cpu::Optimized CPU_Optimized;

    CPU_Optimized* CPU_Optimized_init(
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

        return new CPU_Optimized(P);
    }

    void CPU_Optimized_grid(CPU_Optimized* p,
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

    void CPU_Optimized_degrid(CPU_Optimized* p,
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

    void CPU_Optimized_transform(CPU_Optimized* p,
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

    void CPU_Optimized_destroy(CPU_Optimized* p) {
       delete p;
    }

}  // end extern "C"
