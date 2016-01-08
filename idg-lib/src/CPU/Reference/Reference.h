/**
 *  \class Reference
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_REFERENCE_H_
#define IDG_CPU_REFERENCE_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class Reference : public CPU {

                public:
                    /// Constructors
                    Reference(Parameters params,
                              Compiler compiler = default_compiler(),
                              Compilerflags flags = default_compiler_flags(),
                              ProxyInfo info = default_info());

                    // Disallow assignment and pass-by-value
                    Reference& operator=(const Reference& rhs) = delete;
                    Reference(const Reference& v) = delete;

                    /// Destructor
                    virtual ~Reference() = default;

                    // Get default values for ProxyInfo
                    static ProxyInfo default_info();
                    static std::string default_compiler();
                    static std::string default_compiler_flags();

            }; // class Reference

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
