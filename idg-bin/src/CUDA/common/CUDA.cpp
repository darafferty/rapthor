#include "CUDA.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {

            void PowerRecord::enqueue(cu::Stream &stream) {
                stream.record(event);
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &state);
            }

            void PowerRecord::getPower(CUstream, CUresult, void *userData) {
                *static_cast<PowerSensor::State *>(userData) = powerSensor.read();
            }

            /// Constructors
            CUDA::CUDA(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info,
                int max_nr_timesteps_gridder,
                int max_nr_timesteps_degridder)
              : mInfo(info),
                max_nr_timesteps_gridder(max_nr_timesteps_gridder),
                max_nr_timesteps_degridder(max_nr_timesteps_degridder)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                // Initialize CUDA
                cu::init();
                device = new cu::Device(deviceNumber);
                context = new cu::Context(*device);
                context->setCurrent();

                // Set/check parameters
                mParams = params;
                parameter_sanity_check(); // throws exception if bad parameters

                // Compile kernels
                compile(compiler, flags);
                load_shared_objects();
                find_kernel_functions();

                // Initialize power sensor
                #if defined(MEASURE_POWER_ARDUINO)
                const char *str_power_sensor = getenv("POWER_SENSOR");
                if (!str_power_sensor) str_power_sensor = POWER_SENSOR;
                const char *str_power_file = getenv("POWER_FILE");
                if (!str_power_file) str_power_file = POWER_FILE;
                cout << "Opening power sensor: " << str_power_sensor << endl;
                cout << "Writing power consumption to file: " << str_power_file << endl;
                powerSensor.init(str_power_sensor, str_power_file);
                #else
                powerSensor.init();
                #endif
            }

            CUDA::~CUDA()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // unload shared objects by ~Module
                for (unsigned int i = 0; i < modules.size(); i++) {
                    delete modules[i];
                }

                // Delete .ptx files
                if (mInfo.delete_shared_objects()) {
                    for (auto libname : mInfo.get_lib_names()) {
                        string lib = mInfo.get_path_to_lib() + "/" + libname;
                        remove(lib.c_str());
                    }
                    rmdir(mInfo.get_path_to_lib().c_str());
                }
            }

            string CUDA::make_tempdir() {
                char _tmpdir[] = "/tmp/idg-XXXXXX";
                char *tmpdir = mkdtemp(_tmpdir);
                #if defined(DEBUG)
                cout << "Temporary files will be stored in: " << tmpdir << endl;
                #endif
                return tmpdir;
            }

            string CUDA::default_compiler() {
                return "nvcc";
            }

            string CUDA::default_compiler_flags() {
                #if defined(DEBUG)
                return "-use_fast_math -lineinfo -src-in-ptx";
                #else
                return "-use_fast_math";
                #endif
            }

            /* Sizeof routines */
            uint64_t CUDA::sizeof_subgrids(int nr_subgrids) {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t CUDA::sizeof_uvw(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                return 1ULL * nr_baselines * nr_time * sizeof(UVW);
            }

            uint64_t CUDA::sizeof_visibilities(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                return 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(complex<float>);
            }

            uint64_t CUDA::sizeof_metadata(int nr_subgrids) {
                return 1ULL * nr_subgrids * sizeof(Metadata);
            }

            uint64_t CUDA::sizeof_grid() {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                return 1ULL * nr_polarizations * gridsize * gridsize * sizeof(complex<float>);
            }

            uint64_t CUDA::sizeof_wavenumbers() {
                auto nr_channels = mParams.get_nr_channels();
                return 1ULL * nr_channels * sizeof(float);
            }

            uint64_t CUDA::sizeof_aterm() {
                auto nr_stations = mParams.get_nr_stations();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t CUDA::sizeof_spheroidal() {
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * subgridsize * subgridsize * sizeof(complex<float>);
            }


#if 0
            /* Misc routines */
            int CUDA::get_max_nr_timesteps_gridder() {
                // Get size of shared memory
                cu::Device &device = get_device();
                int smem_bytes = device.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>();

                // Parameters
                auto nr_concurrent_threadblocks = 2;
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();

                // Compute max_nr_timesteps
                int max_nr_timesteps = ((smem_bytes / (nr_concurrent_threadblocks * sizeof(float))) - nr_channels ) /
                                       (nr_channels * nr_polarizations * 2 + 3);

                // Align max_nr_timesteps to the number of threads in a warp
                int nr_threads = 32;
                int nr_warps = max_nr_timesteps / nr_threads;
                max_nr_timesteps = nr_threads * nr_warps;

                return max_nr_timesteps;
            }
#endif


            /* High level routines */
            void CUDA::transform(
                DomainAtoDomainB direction,
                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                // Initialize
                cu::Context &context = get_context();
                cu::HostMemory h_grid(grid, sizeof_grid());

                // Load kernels
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Initialize
                cu::Stream stream;
                context.setCurrent();

                // Performance measurements
                PowerRecord powerRecords[4];

                // Copy grid to device
                cu::DeviceMemory d_grid(sizeof_grid());
                powerRecords[0].enqueue(stream);
                stream.memcpyHtoDAsync(d_grid, h_grid, sizeof_grid());

                // Execute fft
                kernel_fft->plan(gridsize, 1);
                powerRecords[1].enqueue(stream);
                kernel_fft->launch(stream, d_grid, sign);
                powerRecords[2].enqueue(stream);

                // Copy grid to host
                stream.memcpyDtoHAsync(h_grid, d_grid, sizeof_grid());
                powerRecords[3].enqueue(stream);
                stream.synchronize();

                #if defined(REPORT_TOTAL)
                auxiliary::report(" input",
                                  PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state),
                                  0, sizeof_grid(),
                                  PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                auxiliary::report("   fft",
                                  PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                auxiliary::report("output",
                                  PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state),
                                  0, sizeof_grid(),
                                  PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                std::cout << std::endl;
                #endif
            }

            void CUDA::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines,
                                        aterm_offsets, kernel_size,
                                        max_nr_timesteps_gridder);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Gridder> kernel_gridder = get_kernel_gridder();
                unique_ptr<Scaler> kernel_scaler = get_kernel_scaler();
                unique_ptr<Adder> kernel_adder = get_kernel_adder();
                cu::Module *module_fft = (modules[which_module[name_fft]]);

                // Initialize
                cu::Context &context = get_context();
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Host memory
                cu::HostMemory h_visibilities((void *) visibilities, sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw((void *) uvw, sizeof_uvw(nr_baselines));
                cu::HostMemory h_metadata((void *) metadata, sizeof_metadata(nr_subgrids));

                // Device memory
                cu::DeviceMemory d_wavenumbers(sizeof_wavenumbers());
                cu::DeviceMemory d_spheroidal(sizeof_spheroidal());
                cu::DeviceMemory d_aterm(sizeof_aterm());
                cu::DeviceMemory d_grid(sizeof_grid());

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                htodstream.memcpyHtoDAsync(d_aterm, aterm);
                htodstream.synchronize();

                // Performance measurements
                double total_runtime_gridding = 0;
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;
                total_runtime_gridding = -omp_get_wtime();
                PowerSensor::State startState = powerSensor.read();

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                    // Private device memory
                    cu::DeviceMemory d_visibilities(sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(sizeof_uvw(jobsize));

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Private device memory
                        cu::DeviceMemory d_subgrids(sizeof_subgrids(current_nr_subgrids));
                        cu::DeviceMemory d_metadata(sizeof_metadata(current_nr_subgrids));

                        // Power measurement
                        PowerRecord powerRecords[5];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            executestream.waitEvent(outputFree);
                            powerRecords[0].enqueue(executestream);
                            kernel_gridder->launch(
                                executestream, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            executestream.record(inputFree);
                            powerRecords[1].enqueue(executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_nr_subgrids, d_subgrids);
                            powerRecords[3].enqueue(executestream);

                            // Launch adder kernel
                            kernel_adder->launch(
                                executestream, current_nr_subgrids,
                                d_metadata, d_subgrids, d_grid);
                            powerRecords[4].enqueue(executestream);
                            executestream.record(outputFree);
                        }

                        outputFree.synchronize();

                        double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler  = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_adder   = PowerSensor::seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder->flops(current_nr_baselines, current_nr_subgrids),
                                                     kernel_gridder->bytes(current_nr_baselines, current_nr_subgrids),
                                                     PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("    fft", runtime_fft,
                                                     kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                     kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                     PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report(" scaler", runtime_scaler,
                                                     kernel_scaler->flops(current_nr_subgrids),
                                                     kernel_scaler->bytes(current_nr_subgrids),
                                                     PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                        auxiliary::report("  adder", runtime_adder,
                                                     kernel_adder->flops(current_nr_subgrids),
                                                     kernel_adder->bytes(current_nr_subgrids),
                                                     PowerSensor::Watt(powerRecords[3].state, powerRecords[4].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_gridder += runtime_gridder;
                        total_runtime_fft     += runtime_fft;
                        total_runtime_scaler  += runtime_scaler;
                        total_runtime_adder   += runtime_adder;
                        #endif
                    } // end for s
                }

                // Copy grid to host
                dtohstream.synchronize();
                dtohstream.memcpyDtoHAsync(grid, d_grid, sizeof_grid());
                dtohstream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_gridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_scaler   = kernel_scaler->flops(nr_subgrids);
                uint64_t total_bytes_scaler   = kernel_scaler->bytes(nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder->bytes(nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft + total_flops_scaler + total_flops_adder;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft + total_bytes_scaler + total_bytes_adder;
                double   total_watt_gridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }

            void CUDA::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines,
                                        aterm_offsets, kernel_size,
                                        max_nr_timesteps_degridder);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Degridder> kernel_degridder = get_kernel_degridder();
                unique_ptr<Scaler> kernel_scaler = get_kernel_scaler();
                unique_ptr<Splitter> kernel_splitter = get_kernel_splitter();
                cu::Module *module_fft = (modules[which_module[name_fft]]);

                // Initialize
                cu::Context &context = get_context();
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Host memory
                cu::HostMemory h_visibilities((void *) visibilities, sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw((void *) uvw, sizeof_uvw(nr_baselines));
                cu::HostMemory h_metadata((void *) metadata, sizeof_metadata(nr_subgrids));

                // Device memory
                cu::DeviceMemory d_wavenumbers(sizeof_wavenumbers());
                cu::DeviceMemory d_spheroidal(sizeof_spheroidal());
                cu::DeviceMemory d_aterm(sizeof_aterm());
                cu::DeviceMemory d_grid(sizeof_grid());

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                htodstream.memcpyHtoDAsync(d_aterm, aterm);
                htodstream.memcpyHtoDAsync(d_grid, grid);
                htodstream.synchronize();

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_splitter = 0;
                total_runtime_degridding = -omp_get_wtime();
                PowerSensor::State startState = powerSensor.read();

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                    // Private device memory
                    cu::DeviceMemory d_visibilities(sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(sizeof_uvw(jobsize));

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Private device memory
                        cu::DeviceMemory d_subgrids(sizeof_subgrids(current_nr_subgrids));
                        cu::DeviceMemory d_metadata(sizeof_metadata(current_nr_subgrids));

                        // Power measurement
                        PowerRecord powerRecords[5];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                            // Launch splitter kernel
                            executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
                            kernel_splitter->launch(
                                executestream, current_nr_subgrids,
                                d_metadata, d_subgrids, d_grid);
                            powerRecords[1].enqueue(executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            powerRecords[3].enqueue(executestream);
                            kernel_degridder->launch(
                                executestream, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[4].enqueue(executestream);
                            executestream.record(outputReady);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, sizeof_visibilities(current_nr_baselines));
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        double runtime_splitter  = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft       = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_degridder = PowerSensor::seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter->flops(current_nr_subgrids),
                                                       kernel_splitter->bytes(current_nr_subgrids),
                                                       PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("      fft", runtime_fft,
                                                       kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                       kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                       PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder->flops(current_nr_baselines, current_nr_subgrids),
                                                       kernel_degridder->bytes(current_nr_baselines, current_nr_subgrids),
                                                       PowerSensor::Watt(powerRecords[3].state, powerRecords[4].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter  += runtime_splitter;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_degridder += runtime_degridder;
                        #endif
                    } // end for s
                }

                // Wait for all visibilities to be copied to host
                dtohstream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_degridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                uint64_t total_flops_splitter   = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                double   total_watt_degridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }


            void CUDA::compile(Compiler compiler, Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
                string mparameters =  Parameters::definitions(
                  mParams.get_nr_stations(),
                  mParams.get_nr_baselines(),
                  mParams.get_nr_channels(),
                  mParams.get_nr_time(),
                  mParams.get_nr_timeslots(),
                  mParams.get_imagesize(),
                  mParams.get_nr_polarizations(),
                  mParams.get_grid_size(),
                  mParams.get_subgrid_size());

                // Add device capability
                int capability = 10 * device->getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
                                      device->getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
                string compiler_parameters = " -arch=compute_" + to_string(capability) +
                                             " -code=sm_" + to_string(capability);

                stringstream memory_parameters;
                memory_parameters << " -DMAX_NR_TIMESTEPS_GRIDDER=" << max_nr_timesteps_gridder;
                memory_parameters << " -DMAX_NR_TIMESTEPS_DEGRIDDER=" << max_nr_timesteps_degridder;
                string parameters = " " + flags +
                                    " " + compiler_parameters +
                                    " " + mparameters +
                                    " " + memory_parameters.str().c_str();

                vector<string> v = mInfo.get_lib_names();
                #pragma omp parallel for
                for (int i = 0; i < v.size(); i++) {
                    string libname = v[i];
                    // create shared object "libname"
                    string lib = mInfo.get_path_to_lib() + "/" + libname;

                    vector<string> source_files = mInfo.get_source_files(libname);

                    string source;
                    for (auto src : source_files) {
                        source += mInfo.get_path_to_src() + "/" + src + " ";
                    } // source = a.cpp b.cpp c.cpp ...

                    #if defined(DEBUG)
                    cout << lib << " " << source << " " << endl;
                    #endif

                    cu::Source(source.c_str()).compile(lib.c_str(), parameters.c_str());
                } // for each library
            } // compile

            void CUDA::parameter_sanity_check() {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CUDA::load_shared_objects() {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (auto libname : mInfo.get_lib_names()) {
                    string lib = mInfo.get_path_to_lib() + "/" + libname;

                    #if defined(DEBUG)
                    cout << "Loading: " << libname << endl;
                    #endif

                    modules.push_back(new cu::Module(lib.c_str()));
                }
            }


            void CUDA::find_kernel_functions() {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                CUfunction function;
                for (unsigned int i=0; i<modules.size(); i++) {
                    if (cuModuleGetFunction(&function, *modules[i], name_gridder.c_str()) == CUDA_SUCCESS) {
                        // found gridder kernel in module i
                        which_module[name_gridder] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_degridder.c_str()) == CUDA_SUCCESS) {
                        // found degridder kernel in module i
                        which_module[name_degridder] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_fft.c_str()) == CUDA_SUCCESS) {
                        // found fft kernel in module i
                        which_module[name_fft] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_scaler.c_str()) == CUDA_SUCCESS) {
                        // found scaler kernel in module i
                        which_module[name_scaler] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_adder.c_str()) == CUDA_SUCCESS) {
                        // found adder kernel in module i
                        which_module[name_adder] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], name_splitter.c_str()) == CUDA_SUCCESS) {
                        // found adder kernel in module i
                        which_module[name_splitter] = i;
                    }
                } // end for
            } // end find_kernel_functions


         } // namespace cuda
    } // namespace proxy
} // namespace idg
