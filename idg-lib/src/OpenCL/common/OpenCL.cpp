#include "OpenCL.h"

#include "Util.h"
#include "DeviceInstance.h"
#include "Kernels.h"

using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            OpenCL::OpenCL(
                CompileConstants& constants) :
                Proxy(constants)
            {

                #if defined(DEBUG)
                std::cout << "OPENCL::" << __func__ << std::endl;
                #endif

                init_devices();
                print_devices();
                print_compiler_flags();
            }


            OpenCL::~OpenCL()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                free_devices();

                delete context;

				clfftTeardown();
            }

            std::vector<int> OpenCL::compute_jobsize(
                const Plan &plan,
                const unsigned int nr_timesteps,
                const unsigned int nr_channels,
                const unsigned int subgrid_size,
                const unsigned int nr_streams)
            {
                // Read maximum jobsize from environment
                char *cstr_max_jobsize = getenv("MAX_JOBSIZE");
                auto max_jobsize = cstr_max_jobsize ? atoi(cstr_max_jobsize) : 0;

                // Compute the maximum number of subgrids for any baseline
                int max_nr_subgrids = plan.get_max_nr_subgrids();

                // Compute the amount of bytes needed for that job
                auto bytes_required = 0;
                bytes_required += devices[0]->sizeof_visibilities(1, nr_timesteps, nr_channels);
                bytes_required += devices[0]->sizeof_uvw(1, nr_timesteps);
                bytes_required += devices[0]->sizeof_subgrids(max_nr_subgrids, subgrid_size);
                bytes_required += devices[0]->sizeof_metadata(max_nr_subgrids);
                bytes_required *= nr_streams;

                // Adjust jobsize to amount of available device memory
                int nr_devices = devices.size();
                std::vector<int> jobsize(nr_devices);
                for (int i = 0; i < nr_devices; i++) {
                    DeviceInstance *di = devices[i];
				    cl::Device &d = di->get_device();
                    auto bytes_total = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                    jobsize[i] = (bytes_total * 0.8) /  bytes_required;
                    #if defined(DEBUG)
                    printf("Bytes required: %lu\n", bytes_required);
                    printf("Bytes total:    %lu\n", bytes_total);
                    printf("Jobsize: %d\n", jobsize[i]);
                    #endif
                }

                return jobsize;
            } // end compute_jobsize

            void OpenCL::init_devices() {
                // Create context
                context = new cl::Context(CL_DEVICE_TYPE_ALL);

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
                        mConstants, *context, device_numbers[i], power_sensor, power_file);
                    devices.push_back(device);
                }
            }

            void OpenCL::free_devices() {
                for (DeviceInstance *device : devices) {
                    device->~DeviceInstance();
                }
            }

            void OpenCL::print_devices() {
                std::cout << "Devices: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << *device;
                }
                std::cout << std::endl;
            }

            void OpenCL::print_compiler_flags() {
                std::cout << "Compiler flags: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << device->get_compiler_flags() << std::endl;
                }
                std::cout << std::endl;
            }

        } // end namespace opencl
    } // end namespace proxy
} // end namespace idg
