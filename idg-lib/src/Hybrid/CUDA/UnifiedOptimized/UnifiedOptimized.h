#ifndef IDG_HYBRID_UNIFIED_OPTIMIZED_H_
#define IDG_HYBRID_UNIFIED_OPTIMIZED_H_

#include "idg-hybrid-cuda.h"
#include "CUDA/common/CUDA.h"

namespace idg {
    namespace proxy {
        namespace hybrid {

            class UnifiedOptimized : public cuda::Unified {
                public:
                    UnifiedOptimized(
                        ProxyInfo info = default_info());

                    ~UnifiedOptimized();

                    virtual void do_transform(
                        DomainAtoDomainB direction,
                        Array3D<std::complex<float>>& grid) override;

                private:
                    idg::proxy::cpu::CPU* cpuProxy;
                    idg::proxy::cuda::Generic* gpuProxy;

            }; // class UnifiedOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
