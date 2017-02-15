#ifndef IDG_HYBRID_CUDA_H_
#define IDG_HYBRID_CUDA_H_

#include "idg-cpu.h"
#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {
            class HybridCUDA : public Proxy {

                public:
                    HybridCUDA(
                        CompileConstants constants);

                    ~HybridCUDA();
                    
                private:
                    //idg::proxy::cpu::CPU cpu;
                    //ProxyInfo &mInfo;
                    //std::vector<idg::kernel::cuda::InstanceCUDA*> devices;
            }; // class HybridCUDA
        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
