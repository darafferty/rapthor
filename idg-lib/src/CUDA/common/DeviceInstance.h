#ifndef IDG_CUDA_DEVICEINSTANCE_H_
#define IDG_CUDA_DEVICEINSTANCE_H_

namespace idg {
    namespace proxy {
        namespace cuda {
            class DeviceInstance {
                public:
                    DeviceInstance(int device_number);

                protected:
                    //cu::Context &context;
                    //cu::Device *device;
            };
        }
    }
}

#endif
