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
                #if defined(DEBUG)
                std::cout << "CUDA::" << __func__ << std::endl;
                #endif

                // Get additional parameters
                unsigned int nr_baselines = plan.get_nr_baselines();

                // Check if parameters have changed
                bool reset = false;
                if (nr_stations  != m_gridding_state.nr_stations)  { reset = true; };
                if (nr_timeslots != m_gridding_state.nr_timeslots) { reset = true; };
                if (nr_timesteps != m_gridding_state.nr_timesteps) { reset = true; };
                if (nr_channels  != m_gridding_state.nr_channels)  { reset = true; };
                if (subgrid_size != m_gridding_state.subgrid_size) { reset = true; };
                if (grid_size    != m_gridding_state.grid_size)    { reset = true; };

                // Reuse same jobsize if no parameters have changed
                if (!reset) {
                    #if defined(DEBUG)
                    std::clog << "Reuse previous jobsize" << std::endl;
                    #endif
                    return m_gridding_state.jobsize;
                } else {
                    // Reset all memory allocated by devices
                    for (unsigned d = 0; d < get_num_devices(); d++) {
                        InstanceCUDA& device = get_device(d);
                        device.free_host_memory();
                        device.free_device_memory();
                        device.free_fft_plans();
                    }
                }

                // Set parameters
                m_gridding_state.nr_stations  = nr_stations;
                m_gridding_state.nr_timeslots = nr_timeslots;
                m_gridding_state.nr_timesteps = nr_timesteps;
                m_gridding_state.nr_channels  = nr_channels;
                m_gridding_state.subgrid_size = subgrid_size;
                m_gridding_state.grid_size    = grid_size;
                m_gridding_state.nr_baselines = nr_baselines;

                // Print parameters
                #if defined(DEBUG)
                std::cout << "nr_stations  = " << nr_stations  << std::endl;
                std::cout << "nr_timeslots = " << nr_timeslots << std::endl;
                std::cout << "nr_timesteps = " << nr_timesteps << std::endl;
                std::cout << "nr_channels  = " << nr_channels  << std::endl;
                std::cout << "subgrid_size = " << subgrid_size << std::endl;
                std::cout << "grid_size    = " << grid_size    << std::endl;
                std::cout << "nr_baselines = " << nr_baselines << std::endl;
                #endif

                // Read maximum jobsize from environment
                char *cstr_max_jobsize = getenv("MAX_JOBSIZE");
                auto max_jobsize = cstr_max_jobsize ? atoi(cstr_max_jobsize) : 0;
                #if defined(DEBUG)
                std::cout << "max_jobsize  = " << max_jobsize << std::endl;
                #endif

                // Compute the maximum number of subgrids for any baseline
                int max_nr_subgrids_bl = plan.get_max_nr_subgrids();

                // Compute the amount of bytes needed for that job
                size_t bytes_jobs = 0;
                bytes_jobs += auxiliary::sizeof_visibilities(1, nr_timesteps, nr_channels);
                bytes_jobs += auxiliary::sizeof_uvw(1, nr_timesteps);
                bytes_jobs += auxiliary::sizeof_subgrids(max_nr_subgrids_bl, subgrid_size);
                bytes_jobs += auxiliary::sizeof_metadata(max_nr_subgrids_bl);
                bytes_jobs *= nr_streams;

                // Compute the amount of memory needed for data that is identical for all jobs
                size_t bytes_static = 0;
                bytes_static += auxiliary::sizeof_grid(grid_size);
                bytes_static += auxiliary::sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                bytes_static += auxiliary::sizeof_spheroidal(subgrid_size);
                bytes_static += auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
                bytes_static += auxiliary::sizeof_wavenumbers(nr_channels);
                bytes_static += auxiliary::sizeof_avg_aterm_correction(subgrid_size);

                // Print amount of bytes required
                #if defined(DEBUG)
                std::clog << "Bytes required for static data: " << bytes_static << std::endl;
                std::clog << "Bytes required for job data: "    << bytes_jobs << std::endl;
                #endif

                // Adjust jobsize to amount of available device memory
                unsigned nr_devices = devices.size();
                std::vector<int> jobsize(nr_devices);
                std::vector<int> max_nr_subgrids_job(nr_devices);
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

                    // Subtract the space for static memory from the amount of free memory
                    bytes_free -= bytes_static;

                    // Compute jobsize
                    jobsize[i] = (bytes_free * (1 - fraction_reserved)) /  bytes_jobs;
                    jobsize[i] = max_jobsize > 0 ? min(jobsize[i], max_jobsize) : jobsize[i];
                    jobsize[i] = min(jobsize[i], nr_baselines);

                    // Print jobsize
                    #if defined(DEBUG)
                    printf("Jobsize: %d\n", jobsize[i]);
                    #endif

                    // Get maximum number of subgrids for any job
                    max_nr_subgrids_job[i] = plan.get_max_nr_subgrids(0, nr_baselines, jobsize[i]);
                }

                m_gridding_state.jobsize = jobsize;
                m_gridding_state.max_nr_subgrids = max_nr_subgrids_job;

                return jobsize;
            } // end compute_jobsize

            typedef struct {
                void *dst;
                void *src;
                size_t bytes;
            } MemData;

            void copy_memory(CUstream, CUresult, void *userData)
            {
                MemData *data = static_cast<MemData*>(userData);
                char message[80];
                snprintf(message, 80, "memcpy(%p, %p, %zu)", data->dst, data->src, data->bytes);
                cu::Marker marker(message, 0xffff0000);
                marker.start();
                memcpy(data->dst, data->src, data->bytes);
                marker.end();
                delete data;
            }

            void CUDA::enqueue_copy(
                cu::Stream& stream,
                void *dst,
                void *src,
                size_t bytes)
            {
                // Fill MemData struct
                MemData *data = new MemData();
                data->dst     = dst;
                data->src     = src;
                data->bytes   = bytes;

                // Enqueue memory copy
                stream.addCallback((CUstreamCallback) &copy_memory, data);
            } // end enqueue_copy

        } // end namespace cuda
    } // end namespace proxy
} // end namespace idg
