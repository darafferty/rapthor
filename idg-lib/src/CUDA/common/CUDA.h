#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include <vector>
#include <string>
#include <complex>

#include "idg-common.h"


namespace idg {
    namespace proxy {
        namespace cuda {
            class DeviceInstance;

            class CUDA : public Proxy {
                public:
                    CUDA(
                        Parameters params,
                        ProxyInfo info = default_info());
                    ~CUDA();

                public:
                    void print_compiler_flags();
                    void print_devices();
                    std::vector<DeviceInstance*> get_devices();

                protected:
                    void init_devices();
                    static ProxyInfo default_info();

                protected:
                    ProxyInfo &info;
                    std::vector<DeviceInstance*> devices;

                public:
                    uint64_t sizeof_subgrids(int nr_subgrids);
                    uint64_t sizeof_uvw(int nr_baselines);
                    uint64_t sizeof_visibilities(int nr_baselines);
                    uint64_t sizeof_metadata(int nr_subgrids);
                    uint64_t sizeof_grid();
                    uint64_t sizeof_wavenumbers();
                    uint64_t sizeof_aterm();
                    uint64_t sizeof_spheroidal();

                protected:
                    std::vector<int> compute_jobsize(Plan &plan, int nr_streams);
            };
        } // end namespace idg
    } // end namespace proxy
} // end namespace idg

#endif
