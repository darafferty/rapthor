#ifndef IDG_OPENCLNEW_H_
#define IDG_OPENCLNEW_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include "idg-common.h"

#include "Util.h"
#include "DeviceInstance.h"
#include "Kernels.h"

namespace idg {
    namespace proxy {
        namespace opencl {
            class OpenCL : public Proxy {
                public:
                    OpenCL(
                        Parameters params);

                    ~OpenCL();

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
