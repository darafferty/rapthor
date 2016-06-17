#ifndef IDG_CUDA_DEVICEINSTANCE_H_
#define IDG_CUDA_DEVICEINSTANCE_H_

#include "CU.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class DeviceInstance {
                public:
                    DeviceInstance(int device_number);

                    cu::Context& get_context() const { return *context; }
                    cu::Device&  get_device()  const { return *device; }

                protected:
                    cu::Context *context;
                    cu::Device  *device;
            };

            std::ostream& operator<<(std::ostream& os, DeviceInstance &d);
        }
    }
}

#endif
