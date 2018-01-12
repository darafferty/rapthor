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
                        std::vector<std::string> libraries = default_libraries());

                private:
                    static std::vector<std::string> default_libraries();

            }; // class Reference

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
