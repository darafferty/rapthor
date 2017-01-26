#include <iomanip> // setprecision

#include "InstanceCUDA.h"
#include "PowerRecord.h"

using namespace idg::kernel;

namespace idg {
    namespace kernel {
        namespace cuda {

            // Constructor
            InstanceCUDA::InstanceCUDA(
                CompileConstants &constants,
                ProxyInfo &info,
                int device_number,
                const char *str_power_sensor,
                const char *str_power_file) :
                KernelsInstance(constants),
                mInfo(info),
                mModules(5)
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

                // Load kernels
                load_kernels();

                // Initialize power sensor
                init_powersensor(str_power_sensor, str_power_file);
            }

            // Destructor
            InstanceCUDA::~InstanceCUDA() {
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
                for (cu::Module *module : mModules) { delete module; }
                delete function_gridder;
                delete function_degridder;
                delete function_scaler;
                delete function_adder;
                delete function_splitter;
                delete device;
                //delete context;
            }


            /*
                Compilation
            */
            std::string InstanceCUDA::get_compiler_flags() {
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

            void InstanceCUDA::compile_kernels() {
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
                    mModules[i] = new cu::Module(lib.c_str());
                }
            }

            void InstanceCUDA::load_kernels() {
                CUfunction function;
                int found = 0;
                for (int i = 0; i < mModules.size(); i++) {
                    if (cuModuleGetFunction(&function, *mModules[i], name_gridder.c_str()) == CUDA_SUCCESS) {
                        function_gridder = new cu::Function(function); found++;
                    }
                    if (cuModuleGetFunction(&function, *mModules[i], name_degridder.c_str()) == CUDA_SUCCESS) {
                        function_degridder = new cu::Function(function); found++;
                    }
                    if (cuModuleGetFunction(&function, *mModules[i], name_fft.c_str()) == CUDA_SUCCESS) {
                        function_fft = new cu::Function(function); found++;
                    }
                    if (cuModuleGetFunction(&function, *mModules[i], name_scaler.c_str()) == CUDA_SUCCESS) {
                        function_scaler = new cu::Function(function); found++;
                    }
                    if (cuModuleGetFunction(&function, *mModules[i], name_adder.c_str()) == CUDA_SUCCESS) {
                        function_adder = new cu::Function(function); found++;
                    }
                    if (cuModuleGetFunction(&function, *mModules[i], name_splitter.c_str()) == CUDA_SUCCESS) {
                        function_splitter = new cu::Function(function); found++;
                    }
                }

                if (found != mModules.size()) {
                    std::cerr << "Incorrect number of functions found: " << found << " != " << mModules.size() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceCUDA::set_parameters_kepler() {
                block_gridder    = dim3(16, 16);
                block_degridder  = dim3(128);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 32;
                batch_degridder  = block_degridder.x;
            }

            void InstanceCUDA::set_parameters_maxwell() {
                block_gridder    = dim3(128);
                block_degridder  = dim3(128);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 64;
                batch_degridder  = 64;
            }

            void InstanceCUDA::set_parameters_pascal() {
                block_gridder    = dim3(192);
                block_degridder  = dim3(256);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 64;
                batch_degridder  = 64;
            }

            void InstanceCUDA::set_parameters() {
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

            std::ostream& operator<<(std::ostream& os, InstanceCUDA &d) {
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

            void InstanceCUDA::init_powersensor(
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

            PowerSensor::State InstanceCUDA::measure() {
                return powerSensor->read();
            }

            void InstanceCUDA::measure(
                PowerRecord &record, cu::Stream &stream) {
                stream.record(record.event);
                record.sensor = powerSensor;
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &record);
            }

            void InstanceCUDA::launch_gridder(
                int nr_subgrids,
                int grid_size,
                float image_size,
                float w_offset,
                int nr_channels,
                int nr_stations,
                cu::DeviceMemory& d_uvw,
                cu::DeviceMemory& d_wavenumbers,
                cu::DeviceMemory& d_visibilities,
                cu::DeviceMemory& d_spheroidal,
                cu::DeviceMemory& d_aterm,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid)
            {
                const void *parameters[] = {
                    &grid_size, &image_size, &w_offset, &nr_channels, &nr_stations,
                    d_uvw, d_wavenumbers, d_visibilities,
                    d_spheroidal, d_aterm, d_metadata, d_subgrid };

                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_gridder, grid, block_gridder, 0, parameters);
            }

            void InstanceCUDA::launch_degridder(
                int nr_subgrids,
                int grid_size,
                float image_size,
                float w_offset,
                int nr_channels,
                int nr_stations,
                cu::DeviceMemory& d_uvw,
                cu::DeviceMemory& d_wavenumbers,
                cu::DeviceMemory& d_visibilities,
                cu::DeviceMemory& d_spheroidal,
                cu::DeviceMemory& d_aterm,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid)
            {
                const void *parameters[] = {
                    &grid_size, &image_size, &w_offset, &nr_channels, &nr_stations,
                    d_uvw, d_wavenumbers, d_visibilities,
                    d_spheroidal, d_aterm, d_metadata, d_subgrid };

                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_degridder, grid, block_degridder, 0, parameters);
            }

            void InstanceCUDA::plan_fft(
                int size, int batch)
            {
                // Parameters
                int stride = 1;
                int dist = size * size;
                int nr_correlations = mConstants.get_nr_correlations();

                // Plan bulk fft
                if (fft_plan_bulk == NULL && batch > fft_bulk_size)
                {
                    fft_plan_bulk = new cufft::C2C_2D(size, size, stride, dist, fft_bulk_size * nr_correlations);
                }

                // Plan remainder fft
                if (fft_plan_remainder == NULL || size != fft_planned_size || batch != fft_planned_batch)
                {
                    int remainder = batch % fft_bulk_size;
                    if (fft_plan_remainder) {
                        delete fft_plan_remainder;
                    }
                    if (remainder > 0) {
                        fft_plan_remainder = new cufft::C2C_2D(size, size, stride, dist, remainder * nr_correlations);
                    }
                }

                // Store batch size
                fft_planned_batch = batch;
            }

            void InstanceCUDA::launch_fft(
                cu::DeviceMemory& d_data,
                DomainAtoDomainB direction)
            {
                // Initialize
                cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(d_data));
                int s = 0;
                int nr_correlations = mConstants.get_nr_correlations();
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                // Execute bulk ffts (if any)
                if (fft_planned_batch >= fft_bulk_size) {
                    (*fft_plan_bulk).setStream(*executestream);
                    for (; s < (fft_planned_batch - fft_bulk_size)+1; s += fft_bulk_size) {
                        if (fft_planned_batch - s >= fft_bulk_size) {
                            (*fft_plan_bulk).execute(data_ptr, data_ptr, sign);
                            data_ptr += fft_bulk_size * fft_planned_size * fft_planned_size * nr_correlations;
                        }
                    }
                }

                // Execute remainder ffts
                if (s < fft_planned_batch) {
                    (*fft_plan_remainder).setStream(*executestream);
                    (*fft_plan_remainder).execute(data_ptr, data_ptr, sign);
                }

                // Custom FFT kernel is disabled
                //cuFloatComplex *data_ptr = reinterpret_cast<cuFloatComplex *>(static_cast<CUdeviceptr>(data));
                //int nr_polarizations = parameters.get_nr_polarizations();
                //const void *parameters[] = { &data_ptr, &data_ptr, &sign};
                //executestream->launchKernel(function, fft_planned_batch * nr_polarizations, 1, 1,
                //                    blockX, blockY, blockZ, 0, parameters);
            }

            void InstanceCUDA::launch_adder(
                int nr_subgrids,
                int grid_size,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid,
                cu::DeviceMemory& d_grid)
            {
                const void *parameters[] = { &grid_size, d_metadata, d_subgrid, d_grid };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_adder, grid, block_adder, 0, parameters);
            }

            void InstanceCUDA::launch_splitter(
                int nr_subgrids,
                int grid_size,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid,
                cu::DeviceMemory& d_grid)
            {
                const void *parameters[] = { &grid_size, d_metadata, d_subgrid, d_grid };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_splitter, grid, block_splitter, 0, parameters);
            }

            void InstanceCUDA::launch_scaler(
                int nr_subgrids,
                cu::DeviceMemory& d_subgrid)
            {
                const void *parameters[] = { d_subgrid };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_scaler, grid, block_scaler, 0, parameters);
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

           cu::HostMemory& InstanceCUDA::allocate_host_grid(
                unsigned int grid_size)
            {
                auto size = sizeof_grid(grid_size);
                h_grid = allocate_memory(size, h_grid);
                return *h_grid;
            }

           cu::DeviceMemory& InstanceCUDA::allocate_device_grid(
                unsigned int grid_size)
            {
                auto size = sizeof_grid(grid_size);
                d_grid = allocate_memory(size, d_grid);
                return *d_grid;
            }

           cu::HostMemory& InstanceCUDA::allocate_host_visibilities(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int nr_channels)
            {
                auto size = sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                h_visibilities = allocate_memory(size, h_visibilities);
                return *h_visibilities;
            }

           cu::HostMemory& InstanceCUDA::allocate_host_uvw(
                unsigned int nr_baselines,
                unsigned int nr_timesteps)
            {
                auto size = sizeof_uvw(nr_baselines, nr_timesteps);
                h_uvw = allocate_memory(size, h_uvw);
                return *h_uvw;
            }

            cu::DeviceMemory& InstanceCUDA::allocate_device_wavenumbers(
                unsigned int nr_channels)
            {
                auto size = sizeof_wavenumbers(nr_channels);
                d_wavenumbers = allocate_memory(size, d_wavenumbers);
                return *d_wavenumbers;
            }

            cu::DeviceMemory& InstanceCUDA::allocate_device_aterms(
                unsigned int nr_stations,
                unsigned int nr_timeslots,
                unsigned int subgrid_size)
            {
                auto size = sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                d_aterms = allocate_memory(size, d_aterms);
                return *d_aterms;
            }

            cu::DeviceMemory& InstanceCUDA::allocate_device_spheroidal(
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

            cu::HostMemory& InstanceCUDA::reuse_host_grid(
                unsigned int grid_size,
                void *ptr)
            {
                auto size = sizeof_grid(grid_size);
                h_grid = reuse_memory(size, h_grid, ptr);
                return *h_grid;
            }

            cu::HostMemory& InstanceCUDA::reuse_host_visibilities(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int nr_channels,
                void *ptr)
            {
                auto size = sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                h_visibilities = reuse_memory(size, h_visibilities, ptr);
                return *h_visibilities;
            }

            cu::HostMemory& InstanceCUDA::reuse_host_uvw(
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
