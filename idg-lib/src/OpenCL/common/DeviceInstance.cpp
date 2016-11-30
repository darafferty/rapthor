#include "DeviceInstance.h"

using namespace idg::kernel::opencl;


namespace idg {
    namespace proxy {
        namespace opencl {
            DeviceInstance::DeviceInstance(
                Parameters &parameters,
                cl::Context &context,
                int device_number,
                const char *str_power_sensor,
                const char *str_power_file) :
                parameters(parameters),
                info(info)
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
                compile_kernels(context);

                // Initialize power sensor
                init_powersensor(str_power_sensor, str_power_file);
            }

            DeviceInstance::~DeviceInstance() {
                delete device;
                delete executequeue;
                delete htodqueue;
                delete dtohqueue;
                for (cl::Program *program : programs) {
                    delete program;
                }
            }

            unique_ptr<Gridder> DeviceInstance::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new Gridder(*(programs[which_program.at(name_gridder)]), parameters, block_gridder));
            }

            unique_ptr<Degridder> DeviceInstance::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new Degridder(*(programs[which_program.at(name_degridder)]), parameters, block_degridder));
            }

            unique_ptr<GridFFT> DeviceInstance::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFT(parameters));
            }

            unique_ptr<Scaler> DeviceInstance::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new Scaler(*(programs[which_program.at(name_scaler)]), parameters, block_scaler));
            }

            unique_ptr<Adder> DeviceInstance::get_kernel_adder() const {
                return unique_ptr<Adder>(new Adder(*(programs[which_program.at(name_adder)]), parameters, block_adder));
            }

            unique_ptr<Splitter> DeviceInstance::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new Splitter(*(programs[which_program.at(name_splitter)]), parameters, block_splitter));
            }

            void DeviceInstance::set_parameters_default() {
                batch_gridder   = 32;
                batch_degridder = 256;
                block_gridder   = cl::NDRange(256, 1);
                block_degridder = cl::NDRange(batch_degridder, 1);
                block_adder     = cl::NDRange(128, 1);
                block_splitter  = cl::NDRange(128, 1);
                block_scaler    = cl::NDRange(128, 1);
            }

            void DeviceInstance::set_parameters_fiji() {
				// Fiji parameters are default
            }

			void DeviceInstance::set_parameters_hawaii() {
				// TODO
			}

            void DeviceInstance::set_parameters_tahiti() {
				// TODO
            }

            void DeviceInstance::set_parameters() {
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
            }


            std::string DeviceInstance::get_compiler_flags() {
                // Parameter flags
                std::string flags_parameters = Parameters::definitions(
                    parameters.get_nr_polarizations(),
                    parameters.get_subgrid_size());


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

                // Combine flags
                std::string flags = flags_opencl.str() +
                                    flags_device.str() +
                                    flags_parameters;

                return flags;
            }


            void DeviceInstance::compile_kernels(cl::Context &context) {
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
                    cl::Program *program = new cl::Program(context, source);
                    try {
                        // Build the program
                        (*program).build(devices, options.c_str());
                        programs.push_back(program);
                    } catch (cl::Error &error) {
                        std::cerr << "Compilation failed: " << error.what() << std::endl;
                        std::string msg;
                        (*program).getBuildInfo(*device, CL_PROGRAM_BUILD_LOG, &msg);
                        std::cout << msg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                } // for each library

                // Fill which_program structure
                which_program[name_gridder]   = 0;
                which_program[name_degridder] = 1;
                which_program[name_adder]     = 2;
                which_program[name_splitter]  = 3;
                which_program[name_scaler]    = 4;
            }

            std::ostream& operator<<(std::ostream& os, DeviceInstance &di) {
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

            void DeviceInstance::init_powersensor(
                const char *str_power_sensor,
                const char *str_power_file)
            {
                if (str_power_sensor) {
                    std::cout << "Power sensor: " << str_power_sensor << std::endl;
                    if (str_power_file) {
                        std::cout << "Power file:   " << str_power_file << std::endl;
                    }
                    try {
                        powerSensor = new ArduinoPowerSensor(str_power_sensor, str_power_file);
                    } catch (const std::invalid_argument& ia) {
                        std::cerr << "Invalid argument: " << ia.what() << '\n';
                        powerSensor = new DummyPowerSensor();
                    }
                } else {
                    powerSensor = new DummyPowerSensor();
                }
            }

        } // end namespace opencl
    } // end namespace proxy
} // end namespace idg
