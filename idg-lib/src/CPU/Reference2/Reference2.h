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

            }; // class Reference2

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
