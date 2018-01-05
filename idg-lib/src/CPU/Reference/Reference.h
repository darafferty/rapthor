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
                        std::string libdir = "Reference");

            }; // class Reference

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
