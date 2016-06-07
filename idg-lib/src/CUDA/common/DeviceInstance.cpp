#include "DeviceInstance.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            DeviceInstance::DeviceInstance(
                int device_number)
            {

                // Initialize CUDA
                //cu::init();

                // Initialize device
                //int deviceNumber = 0;
                //const char *str_device_number = getenv("CUDA_DEVICE");
                //if (str_device_number) deviceNumber = atoi(str_device_number);
                //printDevices(deviceNumber);
                //device = new cu::Device(deviceNumber);
                //device = new cu::Device(deviceNumber);

                // Initialize context
                //context = new cu::Context(device);
                //context->setCurrent();
            }
        }
    }
}
