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
                ProxyInfo info) :
                mInfo(info) {

                #if defined(DEBUG)
                std::cout << "CUDA::" << __func__ << std::endl;
                #endif

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

                // Create a device instance for every device
                for (unsigned i = 0; i < device_numbers.size(); i++) {
                    InstanceCUDA *device = new InstanceCUDA(
                        mInfo, i, device_numbers[i]);
                    devices.push_back(device);
                }
            }

            void CUDA::free_devices() {
                for (InstanceCUDA *device : devices) {
                    delete device;
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
                std::cout << "CUDA::" << __func__ << std::endl;
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

                return p;
            } // end default_info

            std::vector<int> CUDA::compute_jobsize(
                const Plan &plan,
                const unsigned int nr_stations,
                const unsigned int nr_timeslots,
                const unsigned int nr_timesteps,
                const unsigned int nr_channels,
                const unsigned int subgrid_size,
                const unsigned int nr_streams,
                const unsigned int grid_size,
                const float fraction_reserved)
            {
                // Read maximum jobsize from environment
                char *cstr_max_jobsize = getenv("MAX_JOBSIZE");
                auto max_jobsize = cstr_max_jobsize ? atoi(cstr_max_jobsize) : 0;

                // Compute the maximum number of subgrids for any baseline
                int max_nr_subgrids = plan.get_max_nr_subgrids();

                // Compute the amount of bytes needed for that job
                size_t bytes_jobs = 0;
                bytes_jobs += auxiliary::sizeof_visibilities(1, nr_timesteps, nr_channels);
                bytes_jobs += auxiliary::sizeof_uvw(1, nr_timesteps);
                bytes_jobs += auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
                bytes_jobs += auxiliary::sizeof_metadata(max_nr_subgrids);
                bytes_jobs *= nr_streams;

                // Compute the amount of memory needed for data that is identical for all jobs
                size_t bytes_static = 0;
                bytes_static += auxiliary::sizeof_grid(grid_size);
                bytes_static += auxiliary::sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);

                // Print amount of bytes required
                #if defined(DEBUG)
                std::clog << "Bytes required for static data: " << bytes_static << std::endl;
                std::clog << "Bytes required for job data: "    << bytes_jobs << std::endl;
                #endif

                // Adjust jobsize to amount of available device memory
                unsigned nr_devices = devices.size();
                std::vector<int> jobsize(nr_devices);
                for (unsigned i = 0; i < nr_devices; i++) {
                    InstanceCUDA *device = devices[i];
                    cu::Context &context = device->get_context();
                    context.setCurrent();

                    // Print device number
                    if (nr_devices > 1) {
                        #if defined(DEBUG)
                        std::clog << "GPU " << i << ", ";
                        #endif
                    }

                    // Get amount of memory available on device
                    auto bytes_free = device->get_device().get_total_memory();
                    #if defined(DEBUG)
                    std::clog << "Bytes free: " << bytes_free << std::endl;
                    #endif

                    // Print reserved memory
                    if (fraction_reserved > 0) {
                        #if defined(DEBUG)
                        std::clog << "Bytes reserved: " << (long) (bytes_free * fraction_reserved) << std::endl;
                        #endif
                    }

                    // Check whether the static data and job data fits at all
                    if (bytes_free < (bytes_static + bytes_jobs)) {
                        std::cerr << "Error! Not enough (free) memory on device to continue.";
                        std::cerr << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    // Subtract the space for the grid from the amount of free memory
                    bytes_free -= bytes_static;

                    // Compute actual jobsize
                    jobsize[i] = (bytes_free * (1 - fraction_reserved)) /  bytes_jobs;
                    jobsize[i] = max_jobsize > 0 ? min(jobsize[i], max_jobsize) : jobsize[i];

                    // Print jobsize
                    #if defined(DEBUG)
                    printf("Jobsize: %d\n", jobsize[i]);
                    #endif
                }

                return jobsize;
            } // end compute_jobsize

        } // end namespace cuda
    } // end namespace proxy
} // end namespace idg
