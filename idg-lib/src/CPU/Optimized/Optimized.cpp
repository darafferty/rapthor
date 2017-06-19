#include "Optimized.h"

#include "arch.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Optimized::Optimized(
                CompileConstants constants,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CPU(constants, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            // Runtime compilation
            ProxyInfo Optimized::default_info()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                string srcdir = auxiliary::get_lib_dir() + "/idg-cpu/Optimized";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = kernel::cpu::InstanceCPU::make_tempdir();

                // Create proxy info
                ProxyInfo p = kernel::cpu::InstanceCPU::default_proxyinfo(srcdir, tmpdir);

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
                    flags << "-std=c++11 -Wall -g";
                } else if (relwithdebinfo == IDG_BUILD_TYPE) {
                    flags << "-std=c++11 -O3 -g";
                } else {
                    flags << "-std=c++11 -Wall -O3";
                }

                // Check instruction set support
                bool avx2_supported   = check_4th_gen_intel_core_features();
                bool avx512_supported = has_intel_knl_features();

                // Intel compiler
                stringstream intel_flags;
                intel_flags << " -qopenmp -xHost -mkl=parallel";

                // Flags to make mkl work with Python ctypes
                #if defined(BUILD_WITH_PYTHON)
                intel_flags << " -lmkl_def";
                #endif

                // Flags for specific cases
                if (!avx512_supported && avx2_supported) {
                    intel_flags  << " -DUSE_VML";
                    #if defined(BUILD_WITH_PYTHON)
                    intel_flags << " -lmkl_vml_avx2";
                    intel_flags << " -lmkl_vml_avx";
                    #endif
                }

                // GNU compiler
                stringstream gnu_flags;

                gnu_flags << " -std=c++11 -fopenmp -ffast-math -Wno-unknown-pragmas";
                if (avx512_supported) {
                    gnu_flags << " -mavx512f -mavx512pf -mavx512er -mavx512cd";
                } else if (avx2_supported) {
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

                // Flags for sincos lookup table
                char *cstr_use_lookup = getenv("USE_LOOKUP");
                if (cstr_use_lookup) {
                    flags << " -DUSE_LOOKUP=1";
                    flags << " -DNR_SAMPLES=" << atoi(cstr_use_lookup);
                }

                return flags.str();
            }

        } // namespace cpu
    } // namespace proxy
} // namespace idg


#include "OptimizedC.h"
