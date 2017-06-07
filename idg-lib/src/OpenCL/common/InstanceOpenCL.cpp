#include <clFFT.h>

#include "InstanceOpenCL.h"
#include "PowerRecord.h"

using namespace idg::kernel::opencl;
using namespace powersensor;

namespace idg {
    namespace kernel {
        namespace opencl {

            // Constructor
            InstanceOpenCL::InstanceOpenCL(
                CompileConstants& constants,
                cl::Context& context,
                int device_number,
                const char *str_power_sensor,
                const char *str_power_file) :
                KernelsInstance(constants),
                mContext(context),
                mPrograms(5)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Initialize members
                device       = new cl::Device(context.getInfo<CL_CONTEXT_DEVICES>()[device_number]);
                executequeue = new cl::CommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE);
                htodqueue    = new cl::CommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE);
                dtohqueue    = new cl::CommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE);

                // Set kernel parameters
                set_parameters();

                // Compile kernels
                compile_kernels();

                // Load kernels
                load_kernels();

                // Initialize power sensor
                init_powersensor(str_power_sensor, str_power_file);

                // Kernel specific initialization
                fft_planned = false;
            }

            // Destructor
            InstanceOpenCL::~InstanceOpenCL() {
                delete device;
                delete executequeue;
                delete htodqueue;
                delete dtohqueue;
                if (fft_planned) { clfftDestroyPlan(&fft_plan); }
                for (cl::Program *program : mPrograms) { delete program; }
                delete kernel_gridder;
                delete kernel_degridder;
                delete kernel_adder;
                delete kernel_splitter;
                delete kernel_scaler;
            }

            void InstanceOpenCL::set_parameters_default() {
                batch_gridder   = 32;
                batch_degridder = 192;
                block_gridder   = cl::NDRange(192, 1);
                block_degridder = cl::NDRange(256, 1);
                block_adder     = cl::NDRange(128, 1);
                block_splitter  = cl::NDRange(128, 1);
                block_scaler    = cl::NDRange(128, 1);
            }

            void InstanceOpenCL::set_parameters_fiji() {
				// Fiji parameters are default
            }

			void InstanceOpenCL::set_parameters_hawaii() {
				// TODO
			}

            void InstanceOpenCL::set_parameters_tahiti() {
				// TODO
            }

            void InstanceOpenCL::set_parameters() {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

				set_parameters_default();

				// Get device name
				std::string name = device->getInfo<CL_DEVICE_NAME>();

				// Overide architecture specific parameters
				if (name.compare("Fiji") == 0) {
					set_parameters_fiji();
				} else if (name.compare("Hawaii") == 0) {
					set_parameters_hawaii();
				} else if (name.compare("Tahiti") == 0) {
					set_parameters_tahiti();
				}

                // Override parameters from environment
                char *cstr_batch_size = getenv("BATCHSIZE");
                if (cstr_batch_size) {
                    auto batch_size = atoi(cstr_batch_size);
                    batch_gridder   = batch_size;
                    batch_degridder = batch_size;
                }
                char *cstr_block_size = getenv("BLOCKSIZE");
                if (cstr_block_size) {
                    auto block_size = atoi(cstr_block_size);
                    block_gridder   = cl::NDRange(block_size, 1);
                    block_degridder = cl::NDRange(block_size, 1);
                }
            }


            std::string InstanceOpenCL::get_compiler_flags() {
                // Parameter flags
                std::stringstream flags_constants;
                flags_constants << " -DNR_POLARIZATIONS=" << mConstants.get_nr_correlations();

				// OpenCL specific flags
                std::stringstream flags_opencl;
				flags_opencl << "-cl-fast-relaxed-math";

				// OpenCL 2.0 specific flags
                float opencl_version = get_opencl_version(*device);
                if (opencl_version >= 2.0) {
                    //flags_opencl << " -cl-std=CL2.0";
                    //flags_opencl << " -DUSE_ATOMIC_FETCH_ADD";
                }

				// Device specific flags
				std::stringstream flags_device;
                flags_device << " -DGRIDDER_BATCH_SIZE="   << batch_gridder;
                flags_device << " -DDEGRIDDER_BATCH_SIZE=" << batch_degridder;
                flags_device << " -DGRIDDER_BLOCK_SIZE="   << block_gridder[0];
                flags_device << " -DDEGRIDDER_BLOCK_SIZE=" << block_degridder[0];

                // Combine flags
                std::string flags = flags_opencl.str() +
                                    flags_device.str() +
                                    flags_constants.str();

                return flags;
            }


            void InstanceOpenCL::compile_kernels()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Source directory
                std::string srcdir = auxiliary::get_lib_dir() + "/idg-opencl";

                #if defined(DEBUG)
                std::cout << "Searching for source files in: " << srcdir << std::endl;
                #endif

                // Get compile flags
                Compilerflags flags = get_compiler_flags();

				// Construct build options
				std::string options = flags +
								      " -I " + srcdir;

                // Create vector of devices
                std::vector<cl::Device> devices;
                devices.push_back(*device);

                // Add all kernels to build
                std::vector<std::string> v;
                v.push_back("KernelGridder.cl");
                v.push_back("KernelDegridder.cl");
                v.push_back("KernelAdder.cl");
                v.push_back("KernelSplitter.cl");
                v.push_back("KernelScaler.cl");

                // Build OpenCL programs
                for (int i = 0; i < v.size(); i++) {
                    // Get source filename
                    std::stringstream _source_file_name;
                    _source_file_name << srcdir << "/" << v[i];
                    std::string source_file_name = _source_file_name.str();

                    // Read source from file
                    std::ifstream source_file(source_file_name.c_str());
                    std::string source(std::istreambuf_iterator<char>(source_file),
                                      (std::istreambuf_iterator<char>()));
                    source_file.close();

                    // Print information about compilation
					#if defined(DEBUG)
                    std::cout << "Compiling: " << _source_file_name.str() << std::endl;
					#endif

                    // Create OpenCL program
                    mPrograms[i] = new cl::Program(mContext, source);
                    try {
                        // Build the program
                        mPrograms[i]->build(devices, options.c_str());
                    } catch (cl::Error error) {
                        std::cerr << "Compilation failed: " << error.what() << std::endl;
                        std::string msg;
                        mPrograms[i]->getBuildInfo(*device, CL_PROGRAM_BUILD_LOG, &msg);
                        std::cout << msg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                } // for each library
            } // end compile_kernels


            void InstanceOpenCL::load_kernels() {
                try {
                    kernel_gridder   = new cl::Kernel(*(mPrograms[0]), name_gridder.c_str());
                    kernel_degridder = new cl::Kernel(*(mPrograms[1]), name_degridder.c_str());
                    kernel_adder     = new cl::Kernel(*(mPrograms[2]), name_adder.c_str());
                    kernel_splitter  = new cl::Kernel(*(mPrograms[3]), name_splitter.c_str());
                    kernel_scaler    = new cl::Kernel(*(mPrograms[4]), name_scaler.c_str());
                } catch (cl::Error& error) {
                    std::cerr << "Loading kernels failed: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            } // end load_kernels

            std::ostream& operator<<(std::ostream& os, InstanceOpenCL &di) {
				cl::Device d = di.get_device();

				os << "Device: "		   << d.getInfo<CL_DEVICE_NAME>()    << std::endl;
				os << "Driver version  : " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
				os << "Device version  : " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
				os << "Compute units   : " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
				os << "Clock frequency : " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
				os << "Global memory   : " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() * 1e-9 << " Gb" << std::endl;
				os << "Local memory    : " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() * 1e-6 << " Mb" << std::endl;
				os << std::endl;

                return os;
            }

            void InstanceOpenCL::init_powersensor(
                const char *str_power_sensor,
                const char *str_power_file)
            {
                powerSensor = DummyPowerSensor::create();
                #if defined(HAVE_POWERSENSOR)
                if (use_powersensor(name_arduino, str_power_sensor)) {
                    powerSensor = arduino::ArduinoPowerSensor::create(str_power_sensor, str_power_file);
                }
                #endif
            }

            State InstanceOpenCL::measure() {
                return powerSensor->read();
            }

            void InstanceOpenCL::measure(
                PowerRecord &record,
                cl::CommandQueue &queue)
            {
                record.sensor = powerSensor;
                record.enqueue(queue);
            }

            /*
                Kernels
            */
            void InstanceOpenCL::launch_gridder(
                int nr_timesteps,
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                int nr_channels,
                int nr_stations,
                cl::Buffer& d_uvw,
                cl::Buffer& d_wavenumbers,
                cl::Buffer& d_visibilities,
                cl::Buffer& d_spheroidal,
                cl::Buffer& d_aterm,
                cl::Buffer& d_metadata,
                cl::Buffer& d_subgrid)
            {
                int local_size_x = block_gridder[0];
                int local_size_y = block_gridder[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel_gridder->setArg(0,  grid_size);
                kernel_gridder->setArg(1,  subgrid_size);
                kernel_gridder->setArg(2,  image_size);
                kernel_gridder->setArg(3,  w_step);
                kernel_gridder->setArg(4,  nr_channels);
                kernel_gridder->setArg(5,  nr_stations);
                kernel_gridder->setArg(6,  d_uvw);
                kernel_gridder->setArg(7,  d_wavenumbers);
                kernel_gridder->setArg(8,  d_visibilities);
                kernel_gridder->setArg(9,  d_spheroidal);
                kernel_gridder->setArg(10, d_aterm);
                kernel_gridder->setArg(11, d_metadata);
                kernel_gridder->setArg(12, d_subgrid);
                try {
                    executequeue->enqueueNDRangeKernel(
                        *kernel_gridder, cl::NullRange, global_size, block_gridder);
                } catch (cl::Error &error) {
                    std::cerr << "Error launching gridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceOpenCL::launch_degridder(
                int nr_timesteps,
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                int nr_channels,
                int nr_stations,
                cl::Buffer& d_uvw,
                cl::Buffer& d_wavenumbers,
                cl::Buffer& d_visibilities,
                cl::Buffer& d_spheroidal,
                cl::Buffer& d_aterm,
                cl::Buffer& d_metadata,
                cl::Buffer& d_subgrid)
            {
                int local_size_x = block_degridder[0];
                int local_size_y = block_degridder[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel_degridder->setArg(0,  grid_size);
                kernel_degridder->setArg(1,  subgrid_size);
                kernel_degridder->setArg(2,  image_size);
                kernel_degridder->setArg(3,  w_step);
                kernel_degridder->setArg(4,  nr_channels);
                kernel_degridder->setArg(5,  nr_stations);
                kernel_degridder->setArg(6,  d_uvw);
                kernel_degridder->setArg(7,  d_wavenumbers);
                kernel_degridder->setArg(8,  d_visibilities);
                kernel_degridder->setArg(9,  d_spheroidal);
                kernel_degridder->setArg(10, d_aterm);
                kernel_degridder->setArg(11, d_metadata);
                kernel_degridder->setArg(12, d_subgrid);
                try {
                    executequeue->enqueueNDRangeKernel(
                        *kernel_degridder, cl::NullRange, global_size, block_degridder);
                } catch (cl::Error &error) {
                    std::cerr << "Error launching degridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceOpenCL::plan_fft(
                int size, int batch)
            {
                // Check wheter a new plan has to be created
                if (!fft_planned ||
                    size  != fft_planned_size ||
                    batch != fft_planned_batch) {
                    // Destroy old plan (if any)
                    if (fft_planned) {
                        clfftDestroyPlan(&fft_plan);
                    }

                    // Create new plan
                    size_t lengths[2] = {(size_t) size, (size_t) size};
                    clfftCreateDefaultPlan(&fft_plan, mContext(), CLFFT_2D, lengths);

                    // Set plan parameters
                    clfftSetPlanPrecision(fft_plan, CLFFT_SINGLE);
                    clfftSetLayout(fft_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
                    clfftSetResultLocation(fft_plan, CLFFT_INPLACE);
                    int distance = size*size;
                    clfftSetPlanDistance(fft_plan, distance, distance);
                    clfftSetPlanBatchSize(fft_plan, batch * mConstants.get_nr_correlations());

                    // Update parameters
                    fft_planned_size = size;
                    fft_planned_batch = batch;

                    // Bake plan
                    cl_command_queue *queue = &(*executequeue)();
                    clfftStatus status = clfftBakePlan(fft_plan, 1, queue, NULL, NULL);
                    if (status != CL_SUCCESS) {
                        std::cerr << "Error baking fft plan" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                fft_planned = true;
            }


            void InstanceOpenCL::launch_fft(
                cl::Buffer &d_data,
                DomainAtoDomainB direction)
            {
                clfftDirection sign = (direction == FourierDomainToImageDomain) ? CLFFT_BACKWARD : CLFFT_FORWARD;
                cl_command_queue *queue = &(*executequeue)();
                clfftStatus status = clfftEnqueueTransform(
                    fft_plan, sign, 1, queue, 0, NULL, NULL, &d_data(), NULL, NULL);
                if (status != CL_SUCCESS) {
                    std::cerr << "Error enqueing fft plan" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceOpenCL::launch_adder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                cl::Buffer& d_metadata,
                cl::Buffer& d_subgrid,
                cl::Buffer& d_grid)
            {
                int local_size_x = block_adder[0];
                int local_size_y = block_adder[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel_adder->setArg(0, grid_size);
                kernel_adder->setArg(1, subgrid_size);
                kernel_adder->setArg(2, d_metadata);
                kernel_adder->setArg(3, d_subgrid);
                kernel_adder->setArg(4, d_grid);
                try {
                    executequeue->enqueueNDRangeKernel(
                        *kernel_adder, cl::NullRange, global_size, block_adder);
                } catch (cl::Error &error) {
                    std::cerr << "Error launching adder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceOpenCL::launch_splitter(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                cl::Buffer& d_metadata,
                cl::Buffer& d_subgrid,
                cl::Buffer& d_grid)
            {
                int local_size_x = block_splitter[0];
                int local_size_y = block_splitter[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel_splitter->setArg(0, grid_size);
                kernel_splitter->setArg(1, subgrid_size);
                kernel_splitter->setArg(2, d_metadata);
                kernel_splitter->setArg(3, d_subgrid);
                kernel_splitter->setArg(4, d_grid);
                try {
                    executequeue->enqueueNDRangeKernel(
                        *kernel_splitter, cl::NullRange, global_size, block_splitter);
                } catch (cl::Error &error) {
                    std::cerr << "Error launching splitter: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceOpenCL::launch_scaler(
                int nr_subgrids,
                int subgrid_size,
                cl::Buffer& d_subgrid)
            {
                int local_size_x = block_scaler[0];
                int local_size_y = block_scaler[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel_scaler->setArg(0, subgrid_size);
                kernel_scaler->setArg(1, d_subgrid);
                try {
                    executequeue->enqueueNDRangeKernel(
                        *kernel_scaler, cl::NullRange, global_size, block_scaler);
                } catch (cl::Error &error) {
                    std::cerr << "Error launching scaler: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

        } // end namespace opencl
    } // end namespace kernel
} // end namespace idg
