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
        namespace cuda {

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
            return "nvcc";
        }
        

        string Reference::default_compiler_flags() 
        {
            /* TODO:
            add flags: -arch=compute_capability -code=sm_capability
            where capability is obtained as follows:
            cu::Device device(deviceNumber);
            int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
                                  device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
            */

            #if defined(DEBUG)
            return "-use_fast_math -lineinfo -src-in-ptx";
            #else
            return "-use_fast_math";
            #endif
        }


        } // namespace cuda
    } // namespace proxy
} // namespace idg
