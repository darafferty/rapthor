// TODO: check which include files are really necessary
#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "idg-config.h"
#include "Reference.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

        /// Constructors
        Reference::Reference(
            Parameters params,
            Compiler compiler,
            Compilerflags flags,
            ProxyInfo info)
            : CPU(params, compiler, flags, info)
        {
            #if defined(DEBUG)
            cout << "Reference::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
        }


        ProxyInfo Reference::default_info()
        {
            #if defined(DEBUG)
            cout << "Reference::" << __func__ << endl;
            #endif

            string  srcdir = string(IDG_SOURCE_DIR) 
                + "/src/CPU/Reference/kernels";

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
