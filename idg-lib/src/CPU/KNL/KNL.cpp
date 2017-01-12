#include "KNL.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            KNL::KNL(
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

            ProxyInfo KNL::default_info()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                string srcdir = auxiliary::get_lib_dir() + "/idg-cpu/KNL";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = kernel::cpu::KernelsCPU::make_tempdir();

                // Create proxy info
                ProxyInfo p = kernel::cpu::KernelsCPU::default_proxyinfo(srcdir, tmpdir);

                return p;
            }


            string KNL::default_compiler()
            {
                #if defined(GNU_CXX_COMPILER)
                // TODO: not tested
                return "g++";
                #elif defined(CLANG_CXX_COMPILER)
                // TODO: not tested
                return "clang++";
                #else
                return "icpc";
                #endif
            }


            string KNL::default_compiler_flags()
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

                // Intel compiler
                stringstream intel_flags;
                intel_flags << " -qopenmp -xHost -mkl=parallel";

                // GNU compiler
                stringstream gnu_flags;
                // TODO: not tested
                gnu_flags << " -std=c++11 -fopenmp -march=core-avx2 -ffast-math";

                // Clang compiler
                stringstream clang_flags;
                // TODO: not tested
                clang_flags << " -std=c++11 -fopenmp";

                // MKL
                stringstream mkl_flags;
                #if defined(HAVE_MKL)
                mkl_flags << " -L" << MKL_LIB_DIR;
                mkl_flags << " -lmkl";
                #endif

                // Add compiler specific flags
                stringstream compiler_flags;
                #if defined(GNU_CXX_COMPILER)
                    flags << gnu_flags.str();
                    #if defined(HAVE_MKL)
                    flags << mkl_flags.str();
                    #else
                    flags << " -lfftw3f";
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
