#include "OpenCLNew.h"

using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            OpenCLNew::OpenCLNew(
                Parameters params) {

                #if defined(DEBUG)
                std::cout << "OPENCL::" << __func__ << std::endl;
                std::cout << params;
                #endif

                mParams = params;
                init_devices();
                print_devices();
                print_compiler_flags();
            }

            void OpenCLNew::init_devices() {
                // Get list of all device numbers
                char *char_opencl_device = getenv("OPENCL_DEVICE");
                std::vector<int> device_numbers;
                if (!char_opencl_device) {
                    // Use device 0 if no OpenCL devices were specified
                    device_numbers.push_back(0);
                } else {
                    device_numbers = idg::auxiliary::split_int(char_opencl_device, ",");
                }

                // Get list of all power sensors
                char *char_power_sensor = getenv("POWER_SENSOR");
                std::vector<std::string> power_sensors = idg::auxiliary::split_string(char_power_sensor, ",");

                // Get list of all power files
                char *char_power_file = getenv("POWER_FILE");
                std::vector<std::string> power_files = idg::auxiliary::split_string(char_power_file, ",");

                // Create a device instance for every device
                for (int i = 0; i < device_numbers.size(); i++) {
                    const char *power_sensor = i < power_sensors.size() ? power_sensors[i].c_str() : NULL;
                    const char *power_file = i < power_files.size() ? power_files[i].c_str() : NULL;
                    DeviceInstance *device = new DeviceInstance(
                        mParams, device_numbers[i], power_sensor, power_file);
                    devices.push_back(device);
                }
            }

            void OpenCLNew::print_devices() {
                std::cout << "Devices: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << *device;
                }
                std::cout << std::endl;
            }

            void OpenCLNew::print_compiler_flags() {
                std::cout << "Compiler flags: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << device->get_compiler_flags() << std::endl;
                }
                std::cout << std::endl;
            }
        } // namespace opencl
    } // namespace proxy
} // namespace idg
