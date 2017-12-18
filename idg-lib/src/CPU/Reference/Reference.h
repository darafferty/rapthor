#ifndef IDG_CPU_REFERENCE2_H_
#define IDG_CPU_REFERENCE2_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class Reference : public CPU {
                public:
                    // Constructor
                    Reference(
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    // Default values for runtime compilation
                    static ProxyInfo default_info();
                    static std::string default_compiler();
                    static std::string default_compiler_flags();

            }; // class Reference

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
