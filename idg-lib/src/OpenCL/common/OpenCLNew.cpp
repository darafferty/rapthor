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


            OpenCLNew::~OpenCLNew()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

				clfftTeardown();
            }


            uint64_t OpenCLNew::sizeof_subgrids(int nr_subgrids) {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t OpenCLNew::sizeof_uvw(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                return 1ULL * nr_baselines * nr_time * sizeof(UVW);
            }

            uint64_t OpenCLNew::sizeof_visibilities(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                return 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(complex<float>);
            }

            uint64_t OpenCLNew::sizeof_metadata(int nr_subgrids) {
                return 1ULL * nr_subgrids * sizeof(Metadata);
            }

            uint64_t OpenCLNew::sizeof_grid() {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                return 1ULL * nr_polarizations * gridsize * gridsize * sizeof(complex<float>);
            }

            uint64_t OpenCLNew::sizeof_wavenumbers() {
                auto nr_channels = mParams.get_nr_channels();
                return 1ULL * nr_channels * sizeof(float);
            }

            uint64_t OpenCLNew::sizeof_aterm() {
                auto nr_stations = mParams.get_nr_stations();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t OpenCLNew::sizeof_spheroidal() {
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * subgridsize * subgridsize * sizeof(complex<float>);
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
