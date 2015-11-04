/**
 *  \class Maxwell
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_MAXWELL_H_
#define IDG_CUDA_MAXWELL_H_

#include "CUDA.h"

namespace idg {
    namespace proxy {
        namespace cuda {

            class Maxwell : public CUDA {

            public:
                /// Constructors
                Maxwell(Parameters params,
                          unsigned deviceNumber = 0,
                          Compiler compiler = default_compiler(),
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());

                /// Destructor
                ~Maxwell() = default;

                static ProxyInfo default_info();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);
                static std::string default_compiler();
                static std::string default_compiler_flags();

            /// Low level routines
            protected:
                virtual void run_gridder(CU_GRIDDER_PARAMETERS);
                virtual void run_degridder(CU_DEGRIDDER_PARAMETERS);

            /// Compilation
            protected:
                void find_kernel_functions();

            }; // class Maxwell

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
