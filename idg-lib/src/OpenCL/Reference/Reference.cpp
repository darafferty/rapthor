#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "idg-config.h"
#include "Reference.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace opencl {

            /// Constructors
            Reference::Reference(
                Parameters params,
                cl::Context &context,
                unsigned deviceNumber,
                Compilerflags flags)
                : OpenCL(params, context, deviceNumber, flags)
            {
                #if defined(DEBUG)
                cout << "Reference::" << __func__ << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            string Reference::default_compiler_flags() {
                return OpenCL::default_compiler_flags();
            }

        } // namespace opencl
    } // namespace proxy
} // namespace idg
