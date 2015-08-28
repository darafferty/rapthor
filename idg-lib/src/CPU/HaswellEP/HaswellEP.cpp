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
#include "HaswellEP.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {

    namespace proxy {

        /// Constructors
        HaswellEP::HaswellEP(
            Parameters params,
            Compiler compiler,
            Compilerflags flags,
            ProxyInfo info)
            : CPU(params, compiler, flags, info)
        {
            #if defined(DEBUG)
            cout << "HaswellEP::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
        }

        ProxyInfo HaswellEP::default_info()
        {
            #if defined(DEBUG)
            cout << "HaswellEP::" << __func__ << endl;
            #endif

            string  srcdir = string(IDG_SOURCE_DIR) 
                + "/src/CPU/HaswellEP/kernels";

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
            #if defined(USING_GNU_CXX_COMPILER)
            return "-Wall -O3 -fopenmp -lfftw3f";
            #else 
            return "-Wall -O3 -openmp -mkl -lmkl_avx2 -lmkl_vml_avx2";
            #endif
        }

    } // namespace proxy

} // namespace idg
