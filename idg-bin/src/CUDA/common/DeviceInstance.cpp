#include "DeviceInstance.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            DeviceInstance::DeviceInstance(
                int device_number)
            {
                device = new cu::Device(device_number);
                context = new cu::Context(*device);
            }

            std::ostream& operator<<(std::ostream& os, DeviceInstance &d) {
                os << "Device:           " << d.get_device().getName() << std::endl;
                os << "Device memory   : " << d.get_device().getTotalMem() / (float) (1000*1000*1000) << " Gb" << std::endl;
                os << "Shared memory   : " << d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>() / 1024 << " Kb"<< std::endl;
                os << "Clock frequency : " << d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>() / 1000 << std::endl;
                os << "Capability      : " << d.get_device().getComputeCapability() << std::endl;
                os << std::endl;
                return os;
            }
        }
    }
}
