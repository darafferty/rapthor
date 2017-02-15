#include <string>

#include <cuda.h>
#include <cudaProfiler.h>

#include "CUDA.h"

#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            CUDA::CUDA(
                CompileConstants constants,
                ProxyInfo info) :
                Proxy(constants),
                mInfo(info) {

                #if defined(DEBUG)
                std::cout << "CUDA::" << __func__ << std::endl;
                #endif

                setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
                cu::init();
                init_devices();
                print_devices();
                print_compiler_flags();
                cuProfilerStart();
            };

            CUDA::~CUDA() {
                cuProfilerStop();
                free_devices();
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
                    InstanceCUDA *device = new InstanceCUDA(
                        mConstants, mInfo, device_numbers[i], power_sensor, power_file);
                    devices.push_back(device);
                }
            }

            void CUDA::free_devices() {
                for (InstanceCUDA *device : devices) {
                    device->~InstanceCUDA();
                }
            }

            void CUDA::print_devices() {
                std::cout << "Devices: " << std::endl;
                for (InstanceCUDA *device : devices) {
                    std::cout << *device;
                }
                std::cout << std::endl;
            }

            void CUDA::print_compiler_flags() {
                std::cout << "Compiler flags: " << std::endl;
                for (InstanceCUDA *device : devices) {
                    std::cout << device->get_compiler_flags() << std::endl;
                }
                std::cout << std::endl;
            }

            unsigned int CUDA::get_num_devices() const
            {
                return devices.size();
            }

            InstanceCUDA& CUDA::get_device(unsigned int i) const
            {
                return *(devices[i]);
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
                //std::string libfft = "FFT.ptx";
                std::string libscaler = "Scaler.ptx";
                std::string libadder = "Adder.ptx";
                std::string libsplitter = "Splitter.ptx";

                p.add_lib(libgridder);
                p.add_lib(libdegridder);
                //p.add_lib(libfft);
                p.add_lib(libscaler);
                p.add_lib(libadder);
                p.add_lib(libsplitter);

                p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
                p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
                //p.add_src_file_to_lib(libfft, "KernelFFT.cu");
                p.add_src_file_to_lib(libscaler, "KernelScaler.cu");
                p.add_src_file_to_lib(libadder, "KernelAdder.cu");
                p.add_src_file_to_lib(libsplitter, "KernelSplitter.cu");

                p.set_delete_shared_objects(true);

                return p;
            } // end default_info

            std::vector<int> CUDA::compute_jobsize(
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
                    InstanceCUDA *device = devices[i];
                    cu::Context &context = device->get_context();
                    context.setCurrent();
                    auto bytes_free = device->get_device().get_free_memory();
                    jobsize[i] = (bytes_free * 0.9) /  bytes_required;
                    jobsize[i] = max_jobsize > 0 ? min(jobsize[i], max_jobsize) : jobsize[i];
                    #if defined(DEBUG)
                    printf("Bytes required: %lu\n", bytes_required);
                    printf("Bytes free:     %lu\n", bytes_free);
                    printf("Jobsize: %d\n", jobsize[i]);
                    #endif
                }

                return jobsize;
            } // end compute_jobsize

        } // end namespace cuda
    } // end namespace proxy
} // end namespace idg
