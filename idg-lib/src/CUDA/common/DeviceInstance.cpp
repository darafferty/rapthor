#include <iomanip> // setprecision

#include "DeviceInstance.h"
#include "PowerRecord.h"

using namespace idg::kernel;

namespace idg {
    namespace kernel {
        namespace cuda {

            // Constructor
            DeviceInstance::DeviceInstance(
                CompileConstants &constants,
                ProxyInfo &info,
                int device_number,
                const char *str_power_sensor,
                const char *str_power_file) :
                Kernels(constants),
                mInfo(info)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Initialize members
                device = new cu::Device(device_number);
                context = new cu::Context(*device);
                context->setCurrent();
                executestream  = new cu::Stream();
                htodstream     = new cu::Stream();
                dtohstream     = new cu::Stream();
                h_visibilities = NULL;
                h_uvw          = NULL;
                h_grid         = NULL;
                d_grid         = NULL;
                d_wavenumbers  = NULL;
                d_aterms       = NULL;
                d_spheroidal   = NULL;

                // Set kernel parameters
                set_parameters();

                // Compile kernels
                compile_kernels();

                // Load modules
                load_modules();

                // Initialize power sensor
                init_powersensor(str_power_sensor, str_power_file);
            }

            // Destructor
            DeviceInstance::~DeviceInstance() {
                delete executestream;
                delete htodstream;
                delete dtohstream;
                if (h_visibilities) { h_visibilities->~HostMemory(); }
                if (h_uvw) { h_uvw->~HostMemory(); }
                if (h_grid) { h_grid->~HostMemory(); }
                if (d_grid) { d_grid->~DeviceMemory(); }
                if (d_wavenumbers) { d_wavenumbers->~DeviceMemory(); }
                if (d_aterms) { d_aterms->~DeviceMemory(); }
                if (d_spheroidal) { d_spheroidal->~DeviceMemory(); }
                delete device;
                //delete context;
            }

            // Gridder class
            Gridder::Gridder(
                cu::Module &module,
                const dim3 block) :
                function(module, name_gridder.c_str()),
                block(block) {}

            void Gridder::launch(
                cu::Stream &stream,
                int nr_subgrids,
                int gridsize,
                float imagesize,
                float w_offset,
                int nr_channels,
                int nr_stations,
                cu::DeviceMemory &d_uvw,
                cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities,
                cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm,
                cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid) {

                const void *parameters[] = {
                    &gridsize, &imagesize, &w_offset, &nr_channels, &nr_stations,
                    d_uvw, d_wavenumbers, d_visibilities,
                    d_spheroidal, d_aterm, d_metadata, d_subgrid };

                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }

            // Degridder class
            Degridder::Degridder(
                cu::Module &module,
                const dim3 block) :
                function(module, name_degridder.c_str()),
                block(block) {}

            void Degridder::launch(
                cu::Stream &stream,
                int nr_subgrids,
                int gridsize,
                float imagesize,
                float w_offset,
                int nr_channels,
                int nr_stations,
                cu::DeviceMemory &d_uvw,
                cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities,
                cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm,
                cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid) {

                const void *parameters[] = {
                    &gridsize, &imagesize, &w_offset, &nr_channels, &nr_stations,
                    d_uvw, d_wavenumbers, d_visibilities,
                    d_spheroidal, d_aterm, d_metadata, d_subgrid };

                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }


            // GridFFT class
            GridFFT::GridFFT(
                unsigned int nr_correlations,
                unsigned int size,
                cu::Module &module) :
                nr_correlations(nr_correlations),
                size(size),
                function(module, name_fft.c_str())
            {
                fft_bulk = NULL;
                fft_remainder = NULL;
            }

            GridFFT::~GridFFT()
            {
                if (fft_bulk) {
                    delete fft_bulk;
                }
                if (fft_remainder) {
                    delete fft_remainder;
                }
            }

            void GridFFT::plan(
                unsigned int batch)
            {
                // Parameters
                int stride = 1;
                int dist = size * size;


                // Plan bulk fft
                if (fft_bulk == NULL && batch > bulk_size)
                {
                    fft_bulk = new cufft::C2C_2D(size, size, stride, dist, bulk_size * nr_correlations);
                }

                // Plan remainder fft
                if (fft_remainder == NULL || batch != planned_batch)
                {
                    int remainder = batch % bulk_size;
                    if (fft_remainder) {
                        delete fft_remainder;
                    }
                    if (remainder > 0) {
                        fft_remainder = new cufft::C2C_2D(size, size, stride, dist, remainder * nr_correlations);
                    }
                }

                // Store batch size
                planned_batch = batch;
            }

            void GridFFT::launch(
                cu::Stream &stream,
                cu::DeviceMemory &data,
                int direction)
            {
                // Initialize
                cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(data));
                int s = 0;

                // Execute bulk ffts (if any)
                if (planned_batch >= bulk_size) {
                    (*fft_bulk).setStream(stream);
                    for (; s < (planned_batch - bulk_size)+1; s += bulk_size) {
                        if (planned_batch - s >= bulk_size) {
                            (*fft_bulk).execute(data_ptr, data_ptr, direction);
                            data_ptr += bulk_size * size * size * nr_correlations;
                        }
                    }
                }

                // Execute remainder ffts
                if (s < planned_batch) {
                    (*fft_remainder).setStream(stream);
                    (*fft_remainder).execute(data_ptr, data_ptr, direction);
                }

                // Custom FFT kernel is disabled
                //cuFloatComplex *data_ptr = reinterpret_cast<cuFloatComplex *>(static_cast<CUdeviceptr>(data));
                //int nr_polarizations = parameters.get_nr_polarizations();
                //const void *parameters[] = { &data_ptr, &data_ptr, &direction};
                //stream.launchKernel(function, planned_batch * nr_polarizations, 1, 1,
                //                    blockX, blockY, blockZ, 0, parameters);
            }


            // Adder class
            Adder::Adder(
                cu::Module &module,
                const dim3 block) :
                function(module, name_adder.c_str()),
                block(block) {}

            void Adder::launch(
                cu::Stream &stream,
                int nr_subgrids,
                int gridsize,
                cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid,
                cu::DeviceMemory &d_grid) {
                const void *parameters[] = { &gridsize, d_metadata, d_subgrid, d_grid };
                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }


            // Splitter class
            Splitter::Splitter(
                cu::Module &module,
                const dim3 block) :
                function(module, name_splitter.c_str()),
                block(block) {}

            void Splitter::launch(
                cu::Stream &stream,
                int nr_subgrids,
                int gridsize,
                cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid,
                cu::DeviceMemory &d_grid) {
                const void *parameters[] = { &gridsize, d_metadata, d_subgrid, d_grid };
                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }


            // Scaler class
            Scaler::Scaler(
                cu::Module &module,
                const dim3 block) :
                function(module, name_scaler.c_str()),
                block(block) {}

            void Scaler::launch(
                cu::Stream &stream,
                int nr_subgrids,
                cu::DeviceMemory &d_subgrid) {
                const void *parameters[] = { d_subgrid };
                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }


            /*
                Compilation
            */
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

            void DeviceInstance::measure(
                PowerRecord &record, cu::Stream &stream) {
                stream.record(record.event);
                record.sensor = powerSensor;
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &record);
            }

            template<typename T>
            T* allocate_memory(
                uint64_t size,
                T* ptr)
            {
                if (ptr && size != ptr->size()) {
                    ptr->~T();
                    ptr = new T(size);
                } else if(!ptr) {
                    ptr = new T(size);
                }
                return ptr;
            }

           cu::HostMemory& DeviceInstance::allocate_host_grid(
                unsigned int grid_size)
            {
                auto size = sizeof_grid(grid_size);
                h_grid = allocate_memory(size, h_grid);
                return *h_grid;
            }

           cu::DeviceMemory& DeviceInstance::allocate_device_grid(
                unsigned int grid_size)
            {
                auto size = sizeof_grid(grid_size);
                d_grid = allocate_memory(size, d_grid);
                return *d_grid;
            }

           cu::HostMemory& DeviceInstance::allocate_host_visibilities(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int nr_channels)
            {
                auto size = sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                h_visibilities = allocate_memory(size, h_visibilities);
                return *h_visibilities;
            }

           cu::HostMemory& DeviceInstance::allocate_host_uvw(
                unsigned int nr_baselines,
                unsigned int nr_timesteps)
            {
                auto size = sizeof_uvw(nr_baselines, nr_timesteps);
                h_uvw = allocate_memory(size, h_uvw);
                return *h_uvw;
            }

            cu::DeviceMemory& DeviceInstance::allocate_device_wavenumbers(
                unsigned int nr_channels)
            {
                auto size = sizeof_wavenumbers(nr_channels);
                d_wavenumbers = allocate_memory(size, d_wavenumbers);
                return *d_wavenumbers;
            }

            cu::DeviceMemory& DeviceInstance::allocate_device_aterms(
                unsigned int nr_stations,
                unsigned int nr_timeslots,
                unsigned int subgrid_size)
            {
                auto size = sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                d_aterms = allocate_memory(size, d_aterms);
                return *d_aterms;
            }

            cu::DeviceMemory& DeviceInstance::allocate_device_spheroidal(
                unsigned int subgrid_size)
            {
                auto size = sizeof_spheroidal(subgrid_size);
                d_spheroidal = allocate_memory(size, d_spheroidal);
                return *d_spheroidal;
            }

            template<typename T>
            T* reuse_memory(
                uint64_t size,
                T* memory,
                void *ptr)
            {
                if (memory) {
                    memory->update(ptr, size);
                } else {
                    memory = new T(ptr, size);
                }
                return memory;
            }

            cu::HostMemory& DeviceInstance::reuse_host_grid(
                unsigned int grid_size,
                void *ptr)
            {
                auto size = sizeof_grid(grid_size);
                h_grid = reuse_memory(size, h_grid, ptr);
                return *h_grid;
            }

            cu::HostMemory& DeviceInstance::reuse_host_visibilities(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int nr_channels,
                void *ptr)
            {
                auto size = sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                h_visibilities = reuse_memory(size, h_visibilities, ptr);
                return *h_visibilities;
            }

            cu::HostMemory& DeviceInstance::reuse_host_uvw(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                void *ptr)
            {
                auto size = sizeof_uvw(nr_baselines, nr_timesteps);
                h_uvw = reuse_memory(size, h_uvw, ptr);
                return *h_uvw;
            }

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg
