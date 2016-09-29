/**
 *  \class KNL
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_KNL_H_
#define IDG_CPU_KNL_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

        class KNL : public CPU {

            public:
                /** Construct a KNL (AVX512) implementation object
                    to use the Proxy class.
                    \param params parameter set using Parameters
                    \param compiler specify compiler to use (e.g. "gcc")
                    \param flags specicfy compiler flags to use (e.g. "-Wall -g")
                    \param info specicfy runtime compile settings (advanced setting)
                */
                KNL(Parameters params,
                          Compiler compiler = default_compiler(),
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());

                // Disallow assignment and pass-by-value
                KNL& operator=(const KNL& rhs) = delete;
                KNL(const KNL& v) = delete;

                // Destructor
                virtual ~KNL() = default;

                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();

        }; // class KNL

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
