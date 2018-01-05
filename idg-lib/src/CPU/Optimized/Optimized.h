#ifndef IDG_CPU_OPTIMIZED_H_
#define IDG_CPU_OPTIMIZED_H_

#include "idg-cpu.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class Optimized : public CPU {
                public:
                    // Constructor
                    Optimized(
                        std::string libdir = "Optimized");

            }; // class Optimized

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
