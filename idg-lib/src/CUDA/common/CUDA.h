#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include <vector>
#include <complex>

#include "idg-common.h"
#include "idg-powersensor.h"

namespace idg {
    namespace kernel {
        namespace cuda {
            class DeviceInstance;
        }
    }

    namespace proxy {
        namespace cuda {
            class CUDA : public Proxy {
                public:
                    CUDA(
                        CompileConstants constants,
                        ProxyInfo info);

                    ~CUDA();

                public:
                    void print_compiler_flags();

                    void print_devices();

                    unsigned int get_num_devices() const;
                    idg::kernel::cuda::DeviceInstance& get_device(unsigned int i) const;

                    std::vector<int> compute_jobsize(
                        const Plan &plan,
                        const unsigned int nr_timesteps,
                        const unsigned int nr_channels,
                        const unsigned int subgrid_size,
                        const unsigned int nr_streams);

                protected:
                    void init_devices();
                    void free_devices();
                    static ProxyInfo default_info();

                private:
                    ProxyInfo &mInfo;
                    std::vector<idg::kernel::cuda::DeviceInstance*> devices;

            };
        } // end namespace idg
    } // end namespace proxy
} // end namespace idg

#endif
