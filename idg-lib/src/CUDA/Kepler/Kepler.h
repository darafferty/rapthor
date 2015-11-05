/**
 *  \class Kepler
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
            class Kepler : public CUDA {
                public:
                    /// Constructors
                    Kepler(Parameters params,
                              unsigned deviceNumber = 0,
                              Compiler compiler = default_compiler(),
                              Compilerflags flags = default_compiler_flags(),
                              ProxyInfo info = default_info());

                    /// Destructor
                    ~Kepler() = default;

                    static ProxyInfo default_info();
                    static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

                /// Compilation
                protected:
                    void find_kernel_functions();

            }; // class Kepler
        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
