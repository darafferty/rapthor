#include "CUDA.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            CUDA::CUDA(
                Parameters params,
                ProxyInfo info) :
                info(info) {

                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                cout << params;
                #endif

                mParams = params;
                cu::init();
                init_devices();
                print_devices();
                print_compiler_flags();
            };

            void CUDA::init_devices() {
                // The list of CUDA devices to use are read from the environment
                char *char_cuda_device = getenv("CUDA_DEVICE");

                // Get list of all device numbers
                std::vector<int> device_numbers;

                if (!char_cuda_device) {
                    // Use device 0 if no CUDA devices were specified
                    device_numbers.push_back(0);
                } else if (strlen(char_cuda_device) == 1) {
                    // Just one device number was specified
                    device_numbers.push_back(atoi(char_cuda_device));
                } else {
                    // Split device numbers on comma
                    const char *delimiter = (char *) ",";
                    char *token = strtok(char_cuda_device, delimiter);
                    if (token) device_numbers.push_back(atoi(token));
                    while (token) {
                        token = strtok(NULL, delimiter);
                        if (token) device_numbers.push_back(atoi(token));
                    }
                }

                // Create a device instance for every device
                for (int device_number : device_numbers) {
                    DeviceInstance *device = new DeviceInstance(mParams, info, device_number);
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
                cout << "Generic::" << __func__ << endl;
                #endif

                std::string srcdir = std::string(IDG_INSTALL_DIR)
                    + "/lib/kernels/CUDA/";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                char _tmpdir[] = "/tmp/idg-XXXXXX";
                char *tmpdir = mkdtemp(_tmpdir);
                #if defined(DEBUG)
                cout << "Temporary files will be stored in: " << tmpdir << endl;
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
