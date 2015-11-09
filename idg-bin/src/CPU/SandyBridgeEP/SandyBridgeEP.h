/**
 *  \class SandyBridgeEP
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_SANDYBRIDGEEP_H_
#define IDG_SANDYBRIDGEEP_H_

// TODO: check which include files are really necessary
#include <dlfcn.h>
#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "AbstractProxy.h"
#include "../common/CPU.h"
#include "../common/Kernels.h"

namespace idg {
    namespace proxy {
        namespace cpu {

        class SandyBridgeEP : public CPU {

            public:
                /// Constructors
                SandyBridgeEP(Parameters params,
                          Compiler compiler = default_compiler(),
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());
                
                /// Copy constructor
                ///SandyBridgeEP(const SandyBridgeEP& v) = delete;

                /// Destructor
                virtual ~SandyBridgeEP() = default;
    
                /// Assignment
                SandyBridgeEP& operator=(const SandyBridgeEP& rhs) = delete;

                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
                    
        }; // class SandyBridgeEP
    
        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
