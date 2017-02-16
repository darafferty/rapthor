#include "GenericOptimized.h"

namespace idg {
    namespace proxy {
        namespace hybrid {

            GenericOptimized::GenericOptimized(
                CompileConstants constants)
                : HybridCUDA(new idg::proxy::cpu::Optimized(constants), constants)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif
            }

        } // namespace idg
    } // namespace proxy
} // namespace idg

#include "GenericOptimizedC.h"
