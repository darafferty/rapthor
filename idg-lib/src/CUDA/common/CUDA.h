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
                    std::vector<idg::kernel::cuda::DeviceInstance*> get_devices();
                    std::vector<int> compute_jobsize(Plan &plan, int nr_streams);

                protected:
                    void init_devices();
                    static ProxyInfo default_info();

                protected:
                    ProxyInfo &mInfo;
                    std::vector<idg::kernel::cuda::DeviceInstance*> devices;

                public:
                    uint64_t sizeof_subgrids(int nr_subgrids);
                    uint64_t sizeof_uvw(int nr_baselines);
                    uint64_t sizeof_visibilities(int nr_baselines);
                    uint64_t sizeof_metadata(int nr_subgrids);
                    uint64_t sizeof_grid();
                    uint64_t sizeof_wavenumbers();
                    uint64_t sizeof_aterm();
                    uint64_t sizeof_spheroidal();
            };
        } // end namespace idg
    } // end namespace proxy
} // end namespace idg

#endif
