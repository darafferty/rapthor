// TODO: check which include files are really necessary
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
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

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
            cout << "SandyBridgeEP::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
        }

        ProxyInfo SandyBridgeEP::default_info()
        {
            #if defined(DEBUG)
            cout << "SandyBridgeEP::" << __func__ << endl;
            #endif

            string  srcdir = string(IDG_SOURCE_DIR) 
                + "/src/CPU/SandyBridgeEP/kernels";

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
            #if defined(USING_GNU_CXX_COMPILER)
            return "-Wall -O3 -fopenmp -lfftw3f";
            #else 
            return "-Wall -O3 -openmp -mkl -lmkl_avx -lmkl_vml_avx -march=core-avx-i";
            #endif
        }

        } // namespace cpu
    } // namespace proxy
} // namespace idg
