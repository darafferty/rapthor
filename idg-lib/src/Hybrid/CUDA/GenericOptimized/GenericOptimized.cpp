#include "GenericOptimized.h"

namespace idg {
    namespace proxy {
        namespace cuda {

            GenericOptimized::GenericOptimized()
                : HybridCUDA(new idg::proxy::cpu::Optimized())
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif
            }

        } // namespace idg
    } // namespace cuda
} // namespace idg

#include "GenericOptimizedC.h"
