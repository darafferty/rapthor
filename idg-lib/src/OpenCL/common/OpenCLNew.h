#ifndef IDG_OPENCLNEW_H_
#define IDG_OPENCLNEW_H_

#include "idg-common.h"

#include "Util.h"
#include "DeviceInstance.h"
#include "Kernels.h"

namespace idg {
    namespace proxy {
        namespace opencl {
            class OpenCLNew : public Proxy {
                public:
                    OpenCLNew(
                        Parameters params);

                    ~OpenCLNew();

                public:
                    void print_compiler_flags();
                    void print_devices();
                    std::vector<DeviceInstance*> get_devices();

                protected:
                    void init_devices();

                protected:
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
            };
        } // end namespace idg
    } // end namespace proxy
} // end namespace idg
#endif
