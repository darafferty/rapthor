#include "OpenCL.h"

using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            OpenCL::OpenCL(
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


            OpenCL::~OpenCL()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (DeviceInstance *device : devices) {
                    delete device;
                }

                delete context;

				clfftTeardown();
            }

            std::vector<int> OpenCL::compute_jobsize(Plan &plan, int nr_streams) {
                // Compute the maximum number of subgrids for any baseline
                int max_nr_subgrids = plan.get_max_nr_subgrids();

                // Compute the amount of bytes needed for that job
                auto bytes_required = 0;
                bytes_required += sizeof_visibilities(1);
                bytes_required += sizeof_uvw(1);
                bytes_required += sizeof_subgrids(max_nr_subgrids);
                bytes_required += sizeof_metadata(max_nr_subgrids);
                bytes_required *= nr_streams;

                // Adjust jobsize to amount of available device memory
                int nr_devices = devices.size();
                std::vector<int> jobsize(nr_devices);
                for (int i = 0; i < nr_devices; i++) {
                    DeviceInstance *di = devices[i];
				    cl::Device &d = di->get_device();
                    auto bytes_total = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                    bytes_total -= sizeof_wavenumbers();
                    bytes_total -= sizeof_spheroidal();
                    bytes_total -= sizeof_aterm();
                    bytes_total -= sizeof_grid();
                    jobsize[i] = (bytes_total * 0.8) /  bytes_required;
                    #if defined(DEBUG)
                    printf("Bytes required: %lu\n", bytes_required);
                    printf("Bytes total:    %lu\n", bytes_total);
                    printf("Jobsize: %d\n", jobsize[i]);
                    #endif
                }

                return jobsize;
            }

            uint64_t OpenCL::sizeof_subgrids(int nr_subgrids) {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_uvw(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                return 1ULL * nr_baselines * nr_time * sizeof(UVW);
            }

            uint64_t OpenCL::sizeof_visibilities(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                return 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_metadata(int nr_subgrids) {
                return 1ULL * nr_subgrids * sizeof(Metadata);
            }

            uint64_t OpenCL::sizeof_grid() {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                return 1ULL * nr_polarizations * gridsize * gridsize * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_wavenumbers() {
                auto nr_channels = mParams.get_nr_channels();
                return 1ULL * nr_channels * sizeof(float);
            }

            uint64_t OpenCL::sizeof_aterm() {
                auto nr_stations = mParams.get_nr_stations();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_spheroidal() {
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * subgridsize * subgridsize * sizeof(complex<float>);
            }


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
                        mParams, *context, device_numbers[i], power_sensor, power_file);
                    devices.push_back(device);
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

            std::vector<DeviceInstance*> OpenCL::get_devices() {
                return devices;
            }

        } // namespace opencl
    } // namespace proxy
} // namespace idg
