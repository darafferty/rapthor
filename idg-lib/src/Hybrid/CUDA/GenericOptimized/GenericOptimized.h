#ifndef IDG_HYBRID_GENERIC_OPTIMIZED_H_
#define IDG_HYBRID_GENERIC_OPTIMIZED_H_

#include "idg-hybrid-cuda.h"

namespace idg {
    namespace proxy {
        namespace cuda {

            class GenericOptimized : public hybrid::HybridCUDA {
                public:
                    // Constructor
                    GenericOptimized();

            }; // class GenericOptimized

        } // namespace cuda
    } // namespace proxy
} // namespace idg

#endif
