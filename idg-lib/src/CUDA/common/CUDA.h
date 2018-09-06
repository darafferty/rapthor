#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include <vector>
#include <complex>

#include "idg-common.h"

namespace idg {
    namespace kernel {
        namespace cuda {
            class InstanceCUDA;
        }
    }

    namespace proxy {
        namespace cuda {
            class CUDA : public Proxy {
                public:
                    CUDA(
                        ProxyInfo info);

                    ~CUDA();

                public:
                    void print_compiler_flags();

                    void print_devices();

                    unsigned int get_num_devices() const;
                    idg::kernel::cuda::InstanceCUDA& get_device(unsigned int i) const;

                    std::vector<int> compute_jobsize(
                        const Plan &plan,
                        const unsigned int nr_stations,
                        const unsigned int nr_timeslots,
                        const unsigned int nr_timesteps,
                        const unsigned int nr_channels,
                        const unsigned int subgrid_size,
                        const unsigned int nr_streams,
                        const unsigned int grid_size = 0,
                        const float fraction_reserved = 0.1);

                protected:
                    void init_devices();
                    void free_devices();
                    static ProxyInfo default_info();

                private:
                    ProxyInfo &mInfo;
                    std::vector<idg::kernel::cuda::InstanceCUDA*> devices;

            };
        } // end namespace idg
    } // end namespace proxy
} // end namespace idg

#endif
