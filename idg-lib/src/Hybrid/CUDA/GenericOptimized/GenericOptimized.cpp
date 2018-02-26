#include "GenericOptimized.h"

namespace idg {
    namespace proxy {
        namespace hybrid {

            GenericOptimized::GenericOptimized()
                : HybridCUDA(new idg::proxy::cpu::Optimized())
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#include "GenericOptimizedC.h"
