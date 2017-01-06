/**
 *  \class Reference
 *
 *  \brief Class to use the reference implementation
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_REFERENCE2_H_
#define IDG_CPU_REFERENCE2_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class Reference2 : public CPU2 {
                public:
                    // Constructor
                    Reference2(
                        CompileConstants constants,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    // Default values for runtime compilation
                    static ProxyInfo default_info();
                    static std::string default_compiler();
                    static std::string default_compiler_flags();
#if 0
            class Reference2 : public CPU2 {

                public:
                    /** Construct a reference implementation object
                        to use the Proxy class.
                       \param params parameter set using Parameters
                       \param compiler specify compiler to use (e.g. "gcc")
                       \param flags specicfy compiler flags to use (e.g. "-Wall -g")
                       \param info specicfy runtime compile settings (advanced setting)
                    */
                    Reference(Parameters params,
                              Compiler compiler = default_compiler(),
                              Compilerflags flags = default_compiler_flags(),
                              ProxyInfo info = default_info());

                    // Disallow assignment and pass-by-value
                    Reference& operator=(const Reference& rhs) = delete;
                    Reference(const Reference& v) = delete;

                    // Destructor
                    virtual ~Reference() = default;

                    // Get default values for ProxyInfo
                    static ProxyInfo default_info();
                    static std::string default_compiler();
                    static std::string default_compiler_flags();

#endif
            }; // class Reference2

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
