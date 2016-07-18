/**
 *  \class Optimized
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_OPTIMIZED_H_
#define IDG_CPU_OPTIMIZED_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

        class Optimized : public CPU {

            public:
                /** Construct a optimized (AVX2) implementation object
                    to use the Proxy class.
                    \param params parameter set using Parameters
                    \param compiler specify compiler to use (e.g. "gcc")
                    \param flags specicfy compiler flags to use (e.g. "-Wall -g")
                    \param info specicfy runtime compile settings (advanced setting)
                */
                Optimized(Parameters params,
                          Compiler compiler = default_compiler(),
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());

                // Disallow assignment and pass-by-value
                Optimized& operator=(const Optimized& rhs) = delete;
                Optimized(const Optimized& v) = delete;

                // Destructor
                virtual ~Optimized() = default;

                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();

        }; // class Optimized

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
