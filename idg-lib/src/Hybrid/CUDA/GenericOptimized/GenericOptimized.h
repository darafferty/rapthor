#ifndef IDG_HYBRID_GENERIC_OPTIMIZED_H_
#define IDG_HYBRID_GENERIC_OPTIMIZED_H_

#include "idg-hybrid-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {

            class GenericOptimized : public HybridCUDA {
                public:
                    // Constructor
                    GenericOptimized(
                        CompileConstants constants);

            }; // class GenericOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
