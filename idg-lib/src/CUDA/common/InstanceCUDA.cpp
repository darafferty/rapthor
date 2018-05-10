#include <iomanip> // setprecision

#include "InstanceCUDA.h"
#include "PowerRecord.h"

using namespace idg::kernel;
using namespace powersensor;

#define NR_CORRELATIONS 4

namespace idg {
    namespace kernel {
        namespace cuda {

            // Constructor
            InstanceCUDA::InstanceCUDA(
                ProxyInfo &info,
                int device_nr,
                int device_id) :
                KernelsInstance(),
                mInfo(info),
                mModules(5),
                h_subgrids_(),
                d_visibilities_(),
                d_uvw_(),
                d_subgrids_(),
                d_metadata_()
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Initialize members
                device = new cu::Device(device_id);
                context = new cu::Context(*device);
                context->setCurrent();
                executestream  = new cu::Stream();
                htodstream     = new cu::Stream();
                dtohstream     = new cu::Stream();
                h_visibilities = NULL;
                h_uvw          = NULL;
                h_grid         = NULL;
                d_wavenumbers  = NULL;
                d_aterms       = NULL;
                d_spheroidal   = NULL;
                d_grid         = NULL;
                fft_plan_bulk  = NULL;
                fft_plan_misc  = NULL;

                // Set kernel parameters
                set_parameters();

                // Compile kernels
                compile_kernels();

                // Load kernels
                load_kernels();

                // Initialize power sensor
                powerSensor = get_power_sensor(sensor_device, device_nr);

                // Initialize power records
                for (int i = 0; i < 2; i++) {
                    records_gridder[i] = new PowerRecord();
                    records_degridder[i] = new PowerRecord();
                    records_subgrid_fft[i] = new PowerRecord();
                    records_adder[i] = new PowerRecord();
                    records_splitter[i] = new PowerRecord();
                    records_scaler[i] = new PowerRecord();
                }
            }

            // Destructor
            InstanceCUDA::~InstanceCUDA() {
                context->setCurrent();
                delete executestream;
                delete htodstream;
                delete dtohstream;
                free_device_memory();
                for (cu::Module *module : mModules) { delete module; }
                if (fft_plan_bulk) { delete fft_plan_bulk; }
                if (fft_plan_misc) { delete fft_plan_misc; }
                delete function_gridder;
                delete function_degridder;
                delete function_scaler;
                delete function_adder;
                delete function_splitter;
                for (int i = 0; i < 2; i++) {
                    delete records_gridder[i];
                    delete records_degridder[i];
                    delete records_subgrid_fft[i];
                    delete records_adder[i];
                    delete records_splitter[i];
                    delete records_scaler[i];
                }
                delete device;
                delete context;
                delete powerSensor;
            }


            /*
                Compilation
            */
            std::string InstanceCUDA::get_compiler_flags() {
                // Constants
                std::stringstream flags_constants;
                flags_constants << "-DNR_POLARIZATIONS=" << NR_CORRELATIONS;

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
                flags_device << " -DGRIDDER_BLOCK_SIZE="   << block_gridder.x;
                flags_device << " -DDEGRIDDER_BLOCK_SIZE=" << block_degridder.x;
                flags_device << " -DTILE_SIZE_GRID="       << tile_size_grid;

                // Include flags
                std::stringstream flags_includes;
                flags_includes << "-I" << IDG_INSTALL_DIR << "/include";

                // Combine flags
                std::string flags = " " + flags_cuda.str() +
                                    " " + flags_device.str() +
                                    " " + flags_constants.str() +
                                    " " + flags_includes.str();
                return flags;
            }

            void InstanceCUDA::compile_kernels() {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Get source directory
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

                // Get compiler flags
                std::string flags = get_compiler_flags();

                // Create vector of source filenames
                std::vector<std::string> src;
                src.push_back("KernelGridder.cu");
                src.push_back("KernelDegridder.cu");
                src.push_back("KernelScaler.cu");
                src.push_back("KernelAdder.cu");
                src.push_back("KernelSplitter.cu");

                // Create vector of cubin filenames
                std::vector<std::string> cubin;
                cubin.push_back("Gridder.cubin");
                cubin.push_back("Degridder.cubin");
                cubin.push_back("Scaler.cubin");
                cubin.push_back("Adder.cubin");
                cubin.push_back("Splitter.cubin");

                // Compile all kernels
                #pragma omp parallel for
                for (int i = 0; i < src.size(); i++) {
                    context->setCurrent();

                    // Create a string with the full path to the cubin file "kernel.cubin"
                    std::string lib = mInfo.get_path_to_lib() + "/" + cubin[i];

                    // Create a string for all sources that are combined
                    std::string source = mInfo.get_path_to_src() + "/" + src[i];

                    // Call the compiler
                    cu::Source(source.c_str()).compile(lib.c_str(), flags.c_str());

                    // Set module
                    mModules[i] = new cu::Module(lib.c_str());
                }
            }

            void InstanceCUDA::load_kernels() {
                CUfunction function;
                int found = 0;

                if (cuModuleGetFunction(&function, *mModules[0], name_gridder.c_str()) == CUDA_SUCCESS) {
                    function_gridder = new cu::Function(function); found++;
                }
                if (cuModuleGetFunction(&function, *mModules[1], name_degridder.c_str()) == CUDA_SUCCESS) {
                    function_degridder = new cu::Function(function); found++;
                }
                if (cuModuleGetFunction(&function, *mModules[2], name_scaler.c_str()) == CUDA_SUCCESS) {
                    function_scaler = new cu::Function(function); found++;
                }
                if (cuModuleGetFunction(&function, *mModules[3], name_adder.c_str()) == CUDA_SUCCESS) {
                    function_adder = new cu::Function(function); found++;
                }
                if (cuModuleGetFunction(&function, *mModules[4], name_splitter.c_str()) == CUDA_SUCCESS) {
                    function_splitter = new cu::Function(function); found++;
                }

                if (found != mModules.size()) {
                    std::cerr << "Incorrect number of functions found: " << found << " != " << mModules.size() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void InstanceCUDA::set_parameters_kepler() {
                block_gridder    = dim3(384);
                block_degridder  = dim3(128);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 56;
                batch_degridder  = 128;
                tile_size_grid   = 128;
            }

            void InstanceCUDA::set_parameters_maxwell() {
                block_gridder    = dim3(128);
                block_degridder  = dim3(256);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 256;
                batch_degridder  = 128;
                tile_size_grid   = 128;
            }

            void InstanceCUDA::set_parameters_pascal() {
                block_gridder    = dim3(128);
                block_degridder  = dim3(256);
                block_adder      = dim3(128);
                block_splitter   = dim3(128);
                block_scaler     = dim3(128);
                batch_gridder    = 256;
                batch_degridder  = 128;
                tile_size_grid   = 128;
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
                    block_gridder   = dim3(block_size);
                    block_degridder = dim3(block_size);
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
                auto supports_managed_memory = d.get_device().getAttribute<CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY>();
                os << "\tUnified memory : " << supports_managed_memory << std::endl;
                os << std::endl;
                return os;
            }

           State InstanceCUDA::measure() {
                return powerSensor->read();
            }

            void InstanceCUDA::measure(
                PowerRecord &record, cu::Stream &stream) {
                record.sensor = powerSensor;
                record.enqueue(stream);
            }

            void InstanceCUDA::report_gridder(CUstream, CUresult, void *userData)
            {
                InstanceCUDA *instance = static_cast<InstanceCUDA*>(userData);
                PowerRecord* start = instance->records_gridder[0];
                PowerRecord* end   = instance->records_gridder[1];
                Report *report     = instance->report;
                report->update_gridder(start->state, end->state);
            }

            void InstanceCUDA::report_degridder(CUstream, CUresult, void *userData)
            {
                InstanceCUDA *instance = static_cast<InstanceCUDA*>(userData);
                PowerRecord* start = instance->records_degridder[0];
                PowerRecord* end   = instance->records_degridder[1];
                Report *report     = instance->report;
                report->update_degridder(start->state, end->state);
            }

            void InstanceCUDA::report_adder(CUstream, CUresult, void *userData)
            {
                InstanceCUDA *instance = static_cast<InstanceCUDA*>(userData);
                PowerRecord* start = instance->records_adder[0];
                PowerRecord* end   = instance->records_adder[1];
                Report *report     = instance->report;
                report->update_adder(start->state, end->state);
            }

            void InstanceCUDA::report_splitter(CUstream, CUresult, void *userData)
            {
                InstanceCUDA *instance = static_cast<InstanceCUDA*>(userData);
                PowerRecord* start = instance->records_splitter[0];
                PowerRecord* end   = instance->records_splitter[1];
                Report *report     = instance->report;
                report->update_splitter(start->state, end->state);
            }

            void InstanceCUDA::report_scaler(CUstream, CUresult, void *userData)
            {
                InstanceCUDA *instance = static_cast<InstanceCUDA*>(userData);
                PowerRecord* start = instance->records_scaler[0];
                PowerRecord* end   = instance->records_scaler[1];
                Report *report     = instance->report;
                report->update_scaler(start->state, end->state);
            }

            void InstanceCUDA::report_subgrid_fft(CUstream, CUresult, void *userData)
            {
                InstanceCUDA *instance = static_cast<InstanceCUDA*>(userData);
                PowerRecord* start = instance->records_subgrid_fft[0];
                PowerRecord* end   = instance->records_subgrid_fft[1];
                Report *report     = instance->report;
                report->update_subgrid_fft(start->state, end->state);
            }

            void InstanceCUDA::launch_gridder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
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
                    &grid_size, &subgrid_size, &image_size, &w_step, &nr_channels, &nr_stations,
                    d_uvw, d_wavenumbers, d_visibilities,
                    d_spheroidal, d_aterm, d_metadata, d_subgrid };

                dim3 grid(nr_subgrids);
                measure(*records_gridder[0], *executestream);
                executestream->launchKernel(*function_gridder, grid, block_gridder, 0, parameters);
                measure(*records_gridder[1], *executestream);
                executestream->addCallback((CUstreamCallback) &report_gridder, this);
            }

            void InstanceCUDA::launch_degridder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
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
                    &grid_size, &subgrid_size, &image_size, &w_step, &nr_channels, &nr_stations,
                    d_uvw, d_wavenumbers, d_visibilities,
                    d_spheroidal, d_aterm, d_metadata, d_subgrid };

                dim3 grid(nr_subgrids);
                measure(*records_degridder[0], *executestream);
                executestream->launchKernel(*function_degridder, grid, block_degridder, 0, parameters);
                measure(*records_degridder[1], *executestream);
                executestream->addCallback((CUstreamCallback) &report_degridder, this);
            }

            void InstanceCUDA::plan_fft(
                int size, int batch)
            {
                int stride = 1;
                int dist = size * size;

                // Plan bulk fft
                if (batch > fft_bulk) {
                    if (fft_plan_bulk) {
                        delete fft_plan_bulk;
                    }
                    fft_plan_bulk = new cufft::C2C_2D(
                        size, size, stride, dist,
                        fft_bulk * NR_CORRELATIONS);
                }

                // Plan remainder fft
                int fft_remainder_size = batch % fft_bulk;

                if (fft_remainder_size) {
                    if (fft_plan_misc) {
                        delete fft_plan_misc;
                    }
                    fft_plan_misc = new cufft::C2C_2D(
                        size, size, stride, dist,
                        fft_remainder_size * NR_CORRELATIONS);
                }

                // Store parameters
                fft_size = size;
                fft_batch = batch;
            }

            void InstanceCUDA::launch_fft(
                cu::DeviceMemory& d_data,
                DomainAtoDomainB direction)
            {
                measure(*records_subgrid_fft[0], *executestream);

                cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(d_data));
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                if (fft_plan_bulk) {
                    fft_plan_bulk->setStream(*executestream);
                }

                int s = 0;
                for (; (s + fft_bulk) <= fft_batch; s += fft_bulk) {
                    fft_plan_bulk->execute(data_ptr, data_ptr, sign);
                    data_ptr += fft_size * fft_size * NR_CORRELATIONS * fft_bulk;
                }
                if (s < fft_batch) {
                    fft_plan_misc->setStream(*executestream);
                    fft_plan_misc->execute(data_ptr, data_ptr, sign);
                }

                measure(*records_subgrid_fft[1], *executestream);
                executestream->addCallback((CUstreamCallback) &report_subgrid_fft, this);
            }

             void InstanceCUDA::launch_fft_unified(
                void *data,
                DomainAtoDomainB direction)
            {
                cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(data);
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                if (fft_plan_bulk) {
                    fft_plan_bulk->setStream(*executestream);
                }

                int s = 0;
                for (; (s + fft_bulk) <= fft_batch; s += fft_bulk) {
                    fft_plan_bulk->execute(data_ptr, data_ptr, sign);
                    data_ptr += fft_size * fft_size * NR_CORRELATIONS * fft_bulk;
                }
                if (s < fft_batch) {
                    fft_plan_misc->setStream(*executestream);
                    fft_plan_misc->execute(data_ptr, data_ptr, sign);
                }
            }

            void InstanceCUDA::launch_fft_unified(
                int size,
                int batch,
                Array3D<std::complex<float>>& grid,
                DomainAtoDomainB direction)
            {
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;
                cufft::C2C_1D fft_plan_row(size, 1, 1, 1);
                cufft::C2C_1D fft_plan_col(size, size, 1, 1);

                for (int i = 0; i < batch; i++) {
                    // Execute 1D FFT over all columns
                    for (int col = 0; col < size; col++) {
                        cufftComplex *ptr = (cufftComplex *) grid.data(i, col, 0);
                        fft_plan_row.execute(ptr, ptr, sign);
                    }

                    // Execute 1D FFT over all rows
                    for (int row = 0; row < size; row++) {
                        cufftComplex *ptr = (cufftComplex *) grid.data(i, 0, row);
                        fft_plan_col.execute(ptr, ptr, sign);
                    }
                }
            }

            void InstanceCUDA::launch_adder(
                int nr_subgrids,
                long grid_size,
                int subgrid_size,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid,
                cu::DeviceMemory& d_grid)
            {
                const bool enable_tiling = false;
                const void *parameters[] = { &grid_size, &subgrid_size, d_metadata, d_subgrid, d_grid, &enable_tiling };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_adder, grid, block_adder, 0, parameters);
            }

            void InstanceCUDA::launch_adder_unified(
                int nr_subgrids,
                long grid_size,
                int subgrid_size,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid,
                void *u_grid)
            {
                const bool enable_tiling = true;
                const void *parameters[] = { &grid_size, &subgrid_size, d_metadata, d_subgrid, &u_grid, &enable_tiling };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_adder, grid, block_adder, 0, parameters);
            }

            void InstanceCUDA::launch_splitter(
                int nr_subgrids,
                long grid_size,
                int subgrid_size,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid,
                cu::DeviceMemory& d_grid)
            {
                const bool enable_tiling = false;
                const void *parameters[] = { &grid_size, &subgrid_size, d_metadata, d_subgrid, d_grid, &enable_tiling };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_splitter, grid, block_splitter, 0, parameters);
            }

            void InstanceCUDA::launch_splitter_unified(
                int nr_subgrids,
                long grid_size,
                int subgrid_size,
                cu::DeviceMemory& d_metadata,
                cu::DeviceMemory& d_subgrid,
                void *u_grid)
            {
                const bool enable_tiling = true;
                const void *parameters[] = { &grid_size, &subgrid_size, d_metadata, d_subgrid, &u_grid, &enable_tiling };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_splitter, grid, block_splitter, 0, parameters);
            }

            void InstanceCUDA::launch_scaler(
                int nr_subgrids,
                int subgrid_size,
                cu::DeviceMemory& d_subgrid)
            {
                const void *parameters[] = { &subgrid_size, d_subgrid };
                dim3 grid(nr_subgrids);
                executestream->launchKernel(*function_scaler, grid, block_scaler, 0, parameters);
            }


            /*
             *  Memory management per device
             *      Maintain one memory object per data structure
             */
            template<typename T>
            T* reuse_memory(
                void *ptr,
                uint64_t size,
                T* memory)
            {
                if (!memory) {
                    memory = new T(ptr, size);
                } else if (memory->get() != ptr) {
                    delete memory;
                    memory = new T(ptr, size);
                } else {
                    memory->resize(size);
                }
                return memory;
            }

            template<typename T>
            T* reuse_memory(
                uint64_t size,
                T* memory)
            {
                if (!memory) {
                    memory = new T(size);
                } else {
                    memory->resize(size);
                }
                return memory;
            }

            cu::HostMemory& InstanceCUDA::get_host_visibilities(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int nr_channels,
                void *ptr)
            {
                auto size = auxiliary::sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                h_visibilities = reuse_memory(ptr, size, h_visibilities);
                return *h_visibilities;
            }

            cu::HostMemory& InstanceCUDA::get_host_uvw(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                void *ptr)
            {
                auto size = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
                h_uvw = reuse_memory(ptr, size, h_uvw);
                return *h_uvw;
            }

            cu::HostMemory& InstanceCUDA::get_host_grid(
                unsigned int grid_size,
                void *ptr)
            {
                auto size = auxiliary::sizeof_grid(grid_size);
                h_grid = ptr == NULL ? reuse_memory(size, h_grid) : reuse_memory(ptr, size, h_grid);
                return *h_grid;
            }

            cu::DeviceMemory& InstanceCUDA::get_device_grid(
                unsigned int grid_size)
            {
                auto size = auxiliary::sizeof_grid(grid_size);
                d_grid = reuse_memory(size, d_grid);
                return *d_grid;
            }

            cu::DeviceMemory& InstanceCUDA::get_device_wavenumbers(
                unsigned int nr_channels)
            {
                auto size = auxiliary::sizeof_wavenumbers(nr_channels);
                d_wavenumbers = reuse_memory(size, d_wavenumbers);
                return *d_wavenumbers;
            }

            cu::DeviceMemory& InstanceCUDA::get_device_aterms(
                unsigned int nr_stations,
                unsigned int nr_timeslots,
                unsigned int subgrid_size)
            {
                auto size = auxiliary::sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                d_aterms = reuse_memory(size, d_aterms);
                return *d_aterms;
            }

            cu::DeviceMemory& InstanceCUDA::get_device_spheroidal(
                unsigned int subgrid_size)
            {
                auto size = auxiliary::sizeof_spheroidal(subgrid_size);
                d_spheroidal = reuse_memory(size, d_spheroidal);
                return *d_spheroidal;
            }


            /*
             *  Memory management per stream
             *      Maintain multiple memory objects per structure
             *      Automatically increases the internal vectors for these objects
             */
            template<typename T>
            T* reuse_memory(
                std::vector<T*>& memories,
                unsigned int id,
                uint64_t size)
            {
                T* ptr = NULL;

                if (memories.size() <= id) {
                    ptr = new T(size);

                    memories.push_back(ptr);
                } else {
                    ptr = memories[id];
                }

                ptr->resize(size);

                return ptr;
            }

            cu::HostMemory& InstanceCUDA::get_host_subgrids(
                unsigned int id,
                unsigned int nr_subgrids,
                unsigned int subgrid_size)
            {
                auto size = auxiliary::sizeof_subgrids(nr_subgrids, subgrid_size);
                return *reuse_memory(h_subgrids_, id, size);
            }

            cu::DeviceMemory& InstanceCUDA::get_device_visibilities(
                unsigned int id,
                unsigned int jobsize,
                unsigned int nr_timesteps,
                unsigned int nr_channels)
            {
                auto size = auxiliary::sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
                return *reuse_memory(d_visibilities_, id, size);
            }

            cu::DeviceMemory& InstanceCUDA::get_device_uvw(
                unsigned int id,
                unsigned int jobsize,
                unsigned int nr_timesteps)
            {
                auto size = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
                return *reuse_memory(d_uvw_, id, size);
            }

            cu::DeviceMemory& InstanceCUDA::get_device_subgrids(
                unsigned int id,
                unsigned int nr_subgrids,
                unsigned int subgrid_size)
            {
                auto size = auxiliary::sizeof_subgrids(nr_subgrids, subgrid_size);
                return *reuse_memory(d_subgrids_, id, size);
            }

            cu::DeviceMemory& InstanceCUDA::get_device_metadata(
                unsigned int id,
                unsigned int nr_subgrids)
            {
                auto size = auxiliary::sizeof_metadata(nr_subgrids);
                return *reuse_memory(d_metadata_, id, size);
            }

            /*
             * Device memory destructor
             */
            void InstanceCUDA::free_device_memory() {
                d_visibilities_.clear();
                d_uvw_.clear();
                d_metadata_.clear();
                d_subgrids_.clear();
                if (d_grid != NULL) {
                    delete d_grid;
                    d_grid = NULL;
                }
                if (d_wavenumbers != NULL) {
                    delete d_wavenumbers;
                    d_wavenumbers = NULL;
                }
                if (d_aterms != NULL) {
                    delete d_aterms;
                    d_aterms = NULL;
                }
                if (d_spheroidal != NULL) {
                    delete d_spheroidal;
                    d_spheroidal = NULL;
                }
            }

            /*
             * Reset device
             */
            void InstanceCUDA::reset() {
                delete executestream;
                delete htodstream;
                delete dtohstream;
                context->reset();
                context = new cu::Context(*device);
                context->setCurrent();
                executestream  = new cu::Stream();
                htodstream     = new cu::Stream();
                dtohstream     = new cu::Stream();
            }

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg
