#include <clFFT.h>

#include "OpenCL.h"

#include "Util.h"
#include "InstanceOpenCL.h"

using namespace idg::kernel::opencl;
using namespace powersensor;

namespace idg {
    namespace proxy {
        namespace opencl {

            // Constructor
            OpenCL::OpenCL()
            {

                #if defined(DEBUG)
                std::cout << "OPENCL::" << __func__ << std::endl;
                #endif

                init_devices();
                print_devices();
                print_compiler_flags();
            }


            // Destructor
            OpenCL::~OpenCL()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                free_devices();
				clfftTeardown();
                delete context;
            }

            unsigned int OpenCL::get_num_devices() const
            {
                return devices.size();
            }

            InstanceOpenCL& OpenCL::get_device(unsigned int i) const
            {
                return *(devices[i]);
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
                bytes_required += auxiliary::sizeof_visibilities(1, nr_timesteps, nr_channels);
                bytes_required += auxiliary::sizeof_uvw(1, nr_timesteps);
                bytes_required += auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
                bytes_required += auxiliary::sizeof_metadata(max_nr_subgrids);
                bytes_required *= nr_streams;

                // Adjust jobsize to amount of available device memory
                int nr_devices = devices.size();
                std::vector<int> jobsize(nr_devices);
                for (int i = 0; i < nr_devices; i++) {
                    InstanceOpenCL *di = devices[i];
				    cl::Device &d = di->get_device();
                    auto bytes_total = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                    jobsize[i] = (bytes_total * 0.9) /  bytes_required;
                    jobsize[i] = max_jobsize > 0 ? min(jobsize[i], max_jobsize) : jobsize[i];
                    #if defined(DEBUG)
                    printf("Bytes required: %u\n", bytes_required);
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

                // Create a device instance for every device
                for (unsigned i = 0; i < device_numbers.size(); i++) {
                    InstanceOpenCL *device = new InstanceOpenCL(
                        *context, i, device_numbers[i]);
                    devices.push_back(device);
                }
            }

            void OpenCL::free_devices() {
                for (InstanceOpenCL *device : devices) {
                    device->~InstanceOpenCL();
                }
            }

            void OpenCL::print_devices() {
                std::cout << "Devices: " << std::endl;
                for (InstanceOpenCL *device : devices) {
                    std::cout << *device;
                }
                std::cout << std::endl;
            }

            void OpenCL::print_compiler_flags() {
                std::cout << "Compiler flags: " << std::endl;
                for (InstanceOpenCL *device : devices) {
                    std::cout << device->get_compiler_flags() << std::endl;
                }
                std::cout << std::endl;
            }

        } // end namespace opencl
    } // end namespace proxy
} // end namespace idg
