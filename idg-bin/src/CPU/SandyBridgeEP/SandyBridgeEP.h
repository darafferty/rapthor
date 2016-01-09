/**
 *  \class SandyBridgeEP
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_SANDYBRIDGEEP_H_
#define IDG_CPU_SANDYBRIDGEEP_H_

#include "idg-cpu.h"

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

                // Disallow assignment and pass-by-value
                SandyBridgeEP& operator=(const SandyBridgeEP& rhs) = delete;
                SandyBridgeEP(const SandyBridgeEP& v) = delete;

                /// Destructor
                virtual ~SandyBridgeEP() = default;

                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();

        }; // class SandyBridgeEP

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
