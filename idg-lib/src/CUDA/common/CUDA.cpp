#include <cuda.h>
#include <cudaProfiler.h>

#include "CUDA.h"
#include "DeviceInstance.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            CUDA::CUDA(
                Parameters params,
                ProxyInfo info) :
                info(info) {

                #if defined(DEBUG)
                std::cout << "CUDA::" << __func__ << std::endl;
                std::cout << params;
                #endif

                mParams = params;
                cu::init();
                init_devices();
                print_devices();
                print_compiler_flags();
                cuProfilerStart();
            };

            CUDA::~CUDA() {
                cuProfilerStop();
            }

            void CUDA::init_devices() {
                // Get list of all device numbers
                char *char_cuda_device = getenv("CUDA_DEVICE");
                std::vector<int> device_numbers;
                if (!char_cuda_device) {
                    // Use device 0 if no CUDA devices were specified
                    device_numbers.push_back(0);
                } else {
                    device_numbers = idg::auxiliary::split_int(char_cuda_device, ",");
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
                        mParams, info, device_numbers[i], power_sensor, power_file);
                    devices.push_back(device);
                }
            }

            void CUDA::print_devices() {
                std::cout << "Devices: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << *device;
                }
                std::cout << std::endl;
            }

            void CUDA::print_compiler_flags() {
                std::cout << "Compiler flags: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << device->get_compiler_flags() << std::endl;
                }
                std::cout << std::endl;
            }

            std::vector<DeviceInstance*> CUDA::get_devices() {
                return devices;
            }

            ProxyInfo CUDA::default_info() {
                #if defined(DEBUG)
                std::cout << "Generic::" << __func__ << std::endl;
                #endif

                std::string srcdir = auxiliary::get_lib_dir() + "/idg-cuda";

                #if defined(DEBUG)
                std::cout << "Searching for source files in: " << srcdir << std::endl;
                #endif

                // Create temp directory
                char _tmpdir[] = "/tmp/idg-XXXXXX";
                char *tmpdir = mkdtemp(_tmpdir);
                #if defined(DEBUG)
                std::cout << "Temporary files will be stored in: " << tmpdir << std::endl;
                #endif

                // Create proxy info
                ProxyInfo p;
                p.set_path_to_src(srcdir);
                p.set_path_to_lib(tmpdir);

                std::string libgridder = "Gridder.ptx";
                std::string libdegridder = "Degridder.ptx";
                std::string libfft = "FFT.ptx";
                std::string libscaler = "Scaler.ptx";
                std::string libadder = "Adder.ptx";
                std::string libsplitter = "Splitter.ptx";

                p.add_lib(libgridder);
                p.add_lib(libdegridder);
                p.add_lib(libfft);
                p.add_lib(libscaler);
                p.add_lib(libadder);
                p.add_lib(libsplitter);

                p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
                p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
                p.add_src_file_to_lib(libfft, "KernelFFT.cu");
                p.add_src_file_to_lib(libscaler, "KernelScaler.cu");
                p.add_src_file_to_lib(libadder, "KernelAdder.cu");
                p.add_src_file_to_lib(libsplitter, "KernelSplitter.cu");

                p.set_delete_shared_objects(true);

                return p;
            }

            std::vector<int> CUDA::compute_jobsize(Plan &plan, int nr_streams) {
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
                    DeviceInstance *device = devices[i];
                    cu::Context &context = device->get_context();
                    context.setCurrent();
                    auto bytes_free = device->get_device().get_free_memory();
                    jobsize[i] = (bytes_free * 0.9) /  bytes_required;
                    #if defined(DEBUG)
                    printf("Bytes required: %lu\n", bytes_required);
                    printf("Bytes free:     %lu\n", bytes_free);
                    printf("Jobsize: %d\n", jobsize[i]);
                    #endif
                }

                return jobsize;
            }

            /* Sizeof routines */
            uint64_t CUDA::sizeof_subgrids(int nr_subgrids) {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize * sizeof(std::complex<float>);
            }

            uint64_t CUDA::sizeof_uvw(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                return 1ULL * nr_baselines * nr_time * sizeof(UVW);
            }

            uint64_t CUDA::sizeof_visibilities(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                return 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(std::complex<float>);
            }

            uint64_t CUDA::sizeof_metadata(int nr_subgrids) {
                return 1ULL * nr_subgrids * sizeof(Metadata);
            }

            uint64_t CUDA::sizeof_grid() {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                return 1ULL * nr_polarizations * gridsize * gridsize * sizeof(std::complex<float>);
            }

            uint64_t CUDA::sizeof_wavenumbers() {
                auto nr_channels = mParams.get_nr_channels();
                return 1ULL * nr_channels * sizeof(float);
            }

            uint64_t CUDA::sizeof_aterm() {
                auto nr_stations = mParams.get_nr_stations();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize * sizeof(std::complex<float>);
            }

            uint64_t CUDA::sizeof_spheroidal() {
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * subgridsize * subgridsize * sizeof(float);
            }

        } // end namespace cuda
    } // end namespace proxy
} // end namespace idg
