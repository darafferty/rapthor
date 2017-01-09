#include "Reference2.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Reference2::Reference2(
                CompileConstants constants,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CPU2(constants, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            // Runtime compilation
            ProxyInfo Reference2::default_info()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                string srcdir = auxiliary::get_lib_dir() + "/idg-cpu/Reference";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = CPU2::make_tempdir();

                // Create proxy info
                ProxyInfo p = CPU2::default_proxyinfo(srcdir, tmpdir);

                return p;
            }


            string Reference2::default_compiler()
            {
                #if defined(INTEL_CXX_COMPILER)
                return "icpc";
                #elif defined(CLANG_CXX_COMPILER)
                return "clang++";
                #else
                return "g++";
                #endif
            }


            string Reference2::default_compiler_flags()
            {
                string debug = "Debug";
                string relwithdebinfo = "RelWithDebInfo";

                #if defined(INTEL_CXX_COMPILER)
                // Settings for the intel compiler
                if (debug == IDG_BUILD_TYPE)
                    return "-std=c++11 -Wall -g -DDEBUG -qopenmp -mkl -lmkl_def";
                else if (relwithdebinfo == IDG_BUILD_TYPE)
                    return "-std=c++11 -O3 -qopenmp -g -mkl -lmkl_def";
                else
                    return "-std=c++11 -Wall -O3 -qopenmp -mkl -lmkl_def";
                #else
                // Settings (gcc or clang)
                if (debug == IDG_BUILD_TYPE)
                    return "-std=c++11 -Wall -g -DDEBUG -fopenmp -lfftw3f";
                else if (relwithdebinfo == IDG_BUILD_TYPE)
                    return "-std=c++11 -O3 -g -fopenmp -lfftw3f";
                else
                    return "-std=c++11 -Wall -O3 -fopenmp -lfftw3f";
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
#if 0
    typedef idg::proxy::cpu::Reference CPU_Reference;

    CPU_Reference* CPU_Reference_init(
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

        return new CPU_Reference(P);
    }

    void CPU_Reference_grid(CPU_Reference* p,
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

    void CPU_Reference_degrid(CPU_Reference* p,
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

#endif
}  // end extern "C"
