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
        namespace opencl {

            /// Constructors
            Reference::Reference(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : OpenCL(params, deviceNumber, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << "Reference::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

           ProxyInfo Reference::default_info() {
               return OpenCL::default_info();
           }

           string Reference::default_compiler() {
                return OpenCL::default_compiler();
           }

           string Reference::default_compiler_flags() {
               return OpenCL::default_compiler_flags();

           }

        } // namespace opencl
    } // namespace proxy
} // namespace idg
