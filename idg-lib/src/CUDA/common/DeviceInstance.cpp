#include "DeviceInstance.h"

using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            DeviceInstance::DeviceInstance(
                CompileConstants &constants,
                ProxyInfo &info,
                int device_number,
                const char *str_power_sensor,
                const char *str_power_file) :
                mConstants(constants),
                mInfo(info)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Initialize members
                device = new cu::Device(device_number);
                context = new cu::Context(*device);
                context->setCurrent();
                executestream = new cu::Stream();
                htodstream = new cu::Stream();
                dtohstream = new cu::Stream();

                // Set kernel parameters
                set_parameters();

                // Compile kernels
                compile_kernels();

                // Load modules
                load_modules();

                // Initialize power sensor
                init_powersensor(str_power_sensor, str_power_file);
            }

            std::string DeviceInstance::get_compiler_flags() {
                // Constants
                std::stringstream flags_constants;
                flags_constants << "-DNR_POLARIZATIONS=" << mConstants.get_nr_correlations();
                flags_constants << " -DSUBGRIDSIZE=" << mConstants.get_subgrid_size();

                // CUDA specific flags
                std::stringstream flags_cuda;
                flags_cuda << "-use_fast_math ";
                flags_cuda << "-lineinfo ";
                flags_cuda << "-src-in-ptx";

                // Device specific flags
                int capability = (*device).get_capability();
                std::stringstream flags_device;
                flags_device << "-arch=sm_"                << capability;
                flags_device << " -DGRIDDER_BATCH_SIZE="   << batch_gridder;
                flags_device << " -DDEGRIDDER_BATCH_SIZE=" << batch_degridder;

                // Combine flags
                std::string flags = " " + flags_cuda.str() +
                                    " " + flags_device.str() +
                                    " " + flags_constants.str();
                return flags;
            }

            void DeviceInstance::compile_kernels() {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Get compiler flags
                std::string flags = get_compiler_flags();

                // Compile all libraries (ptx files)
                std::vector<std::string> v = mInfo.get_lib_names();
                #if !defined(DEBUG)
                #pragma omp parallel for
                #endif
                for (int i = 0; i < v.size(); i++) {
                    context->setCurrent();

                    // Create a string with the full path to the ptx file "libname.ptx"
                    std::string libname = v[i];
                    std::string lib = mInfo.get_path_to_lib() + "/" + libname;

                    // Create a string for all sources that are combined
                    std::vector<std::string> source_files = mInfo.get_source_files(libname);

                    std::string source;
                    for (auto src : source_files) {
                        source += mInfo.get_path_to_src() + "/" + src + " ";
                    } // source = a.cpp b.cpp c.cpp ...

                    #if defined(DEBUG)
                    #pragma omp critical(cout)
                    std::cout << lib << " " << source << " " << std::endl;
                    #endif

                    // Call the compiler
                    cu::Source(source.c_str()).compile(lib.c_str(), flags.c_str());

                    // Set module
                    modules.push_back(new cu::Module(lib.c_str()));
                }
            }

            void DeviceInstance::load_modules() {
                CUfunction function;
                int found = 0;
                for (int i = 0; i < modules.size(); i++) {
                    if (cuModuleGetFunction(&function, *modules[i], name_gridder.c_str()) == CUDA_SUCCESS) {
                        which_module[name_gridder] = i; found++;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_degridder.c_str()) == CUDA_SUCCESS) {
                        which_module[name_degridder] = i; found++;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_fft.c_str()) == CUDA_SUCCESS) {
                        which_module[name_fft] = i; found++;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_scaler.c_str()) == CUDA_SUCCESS) {
                        which_module[name_scaler] = i; found++;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_adder.c_str()) == CUDA_SUCCESS) {
                        which_module[name_adder] = i; found++;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_splitter.c_str()) == CUDA_SUCCESS) {
                        which_module[name_splitter] = i; found++;
                    }
                }

                if (found != modules.size()) {
                    std::cerr << "Incorrect number of modules found: " << found << " != " << modules.size() << std::endl;
                }
            }

            void DeviceInstance::set_parameters_kepler() {
                block_gridder    = dim3(16, 16);
                block_degridder  = dim3(128);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 32;
                batch_degridder  = block_degridder.x;
            }

            void DeviceInstance::set_parameters_maxwell() {
                block_gridder    = dim3(128);
                block_degridder  = dim3(128);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 64;
                batch_degridder  = 64;
            }

            void DeviceInstance::set_parameters_pascal() {
                block_gridder    = dim3(192);
                block_degridder  = dim3(256);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 64;
                batch_degridder  = 64;
            }

            void DeviceInstance::set_parameters() {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                int capability = (*device).get_capability();

                if (capability >= 60) {
                    set_parameters_pascal();
                } else if (capability >= 50) {
                    set_parameters_maxwell();
                } else if (capability >= 30) {
                    set_parameters_kepler();
                } else {
                    // IDG has never been tested with pre-Kepler GPUs
                    set_parameters_kepler();
                }
            }

            std::ostream& operator<<(std::ostream& os, DeviceInstance &d) {
                os << d.get_device().get_name() << std::endl;
                os << std::setprecision(2);
                d.get_context().setCurrent();
                auto device_memory   = d.get_device().get_total_memory(); // Bytes
                auto shared_memory   = d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>(); // Bytes
                auto clock_frequency = d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>() / 1000; // Mhz
                auto mem_frequency   = d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE>() / 1000; // Mhz
                auto nr_sm           = d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>();
                auto mem_bus_width   = d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH>(); // Bits
                os << "\tDevice memory : " << device_memory / (float) (1024*1024*1024) << " Gb" << std::endl;
                os << "\tShared memory : " << shared_memory / (float) 1024 << " Kb"<< std::endl;
                os << "\tClk frequency : " << clock_frequency << " Ghz" << std::endl;
                os << "\tMem frequency : " << mem_frequency << " Ghz" << std::endl;
                os << "\tNumber of SM  : " << nr_sm << std::endl;
                os << "\tMem bus width : " << mem_bus_width << " bit" << std::endl;
                os << "\tMem bandwidth : " << 2 * (mem_bus_width / 8) * mem_frequency / 1000 << " GB/s" << std::endl;
                os << "\tCapability    : " << d.get_device().get_capability() << std::endl;
                os << std::endl;
                return os;
            }

            std::unique_ptr<Gridder> DeviceInstance::get_kernel_gridder() const {
                return std::unique_ptr<Gridder>(new Gridder(
                    *(modules[which_module.at(name_gridder)]), block_gridder));
            }

            std::unique_ptr<Degridder> DeviceInstance::get_kernel_degridder() const {
                return std::unique_ptr<Degridder>(new Degridder(
                    *(modules[which_module.at(name_degridder)]), block_degridder));
            }

            std::unique_ptr<GridFFT> DeviceInstance::get_kernel_fft(unsigned int size) const {
                return std::unique_ptr<GridFFT>(new GridFFT(
                    mConstants.get_nr_correlations(),
                    size,
                    *(modules[which_module.at(name_fft)])));
            }

            std::unique_ptr<Adder> DeviceInstance::get_kernel_adder() const {
                return std::unique_ptr<Adder>(new Adder(
                    *(modules[which_module.at(name_adder)]), block_adder));
            }

            std::unique_ptr<Splitter> DeviceInstance::get_kernel_splitter() const {
                return std::unique_ptr<Splitter>(new Splitter(
                    *(modules[which_module.at(name_splitter)]), block_splitter));
            }

            std::unique_ptr<Scaler> DeviceInstance::get_kernel_scaler() const {
                return std::unique_ptr<Scaler>(new Scaler(
                    *(modules[which_module.at(name_scaler)]), block_scaler));
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
                    powerSensor = new ArduinoPowerSensor(str_power_sensor, str_power_file);
                } else {
                    powerSensor = new DummyPowerSensor();
                }
            }

            PowerSensor::State DeviceInstance::measure() {
                return powerSensor->read();
            }

            void DeviceInstance::measure(PowerRecord &record, cu::Stream &stream) {
                stream.record(record.event);
                record.sensor = powerSensor;
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &record);
            }

        } // end namespace cuda
    } // end namespace proxy
} // end namespace idg
