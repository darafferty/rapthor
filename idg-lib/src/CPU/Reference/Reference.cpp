#include "Reference.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Reference::Reference(
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
            ProxyInfo Reference::default_info()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                string srcdir = auxiliary::get_lib_dir() + "/idg-cpu/Reference";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = kernel::cpu::InstanceCPU::make_tempdir();

                // Create proxy info
                ProxyInfo p = kernel::cpu::InstanceCPU::default_proxyinfo(srcdir, tmpdir);

                return p;
            }


            string Reference::default_compiler()
            {
                #if defined(INTEL_CXX_COMPILER)
                return "icpc";
                #elif defined(CLANG_CXX_COMPILER)
                return "clang++";
                #else
                return "g++";
                #endif
            }


            string Reference::default_compiler_flags()
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

#include "ReferenceC.h"
