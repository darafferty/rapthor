#ifndef IDG_CPU_KNL_H_
#define IDG_CPU_KNL_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class KNL : public CPU {
                public:
                    // Constructor
                    KNL(
                        CompileConstants constants,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    // Default values for runtime compilation
                    static ProxyInfo default_info();
                    static std::string default_compiler();
                    static std::string default_compiler_flags();

            }; // class KNL

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
