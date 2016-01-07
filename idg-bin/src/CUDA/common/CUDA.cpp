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
                ProxyInfo info)
              : mInfo(info)
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

            /* High level routines */
            void CUDA::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();

                cu::Context &context = get_context();
                cu::HostMemory h_grid(grid, SIZEOF_GRID);
                transform(direction, context, h_grid);
            }

            void CUDA::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const complex<float> *aterm,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Initialize metadata
                vector<Metadata> metadata = init_metadata(uvw, wavenumbers, baselines);
                auto nr_subgrids = metadata.size();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
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
                cu::HostMemory h_visibilities((void *) visibilities, nr_subgrids * SIZEOF_VISIBILITIES);
                cu::HostMemory h_uvw((void *) uvw, nr_subgrids * SIZEOF_UVW);
                cu::HostMemory h_metadata((void *) metadata.data(), nr_subgrids * SIZEOF_METADATA);

                // Device memory
                cu::DeviceMemory d_wavenumbers(SIZEOF_WAVENUMBERS);
                cu::DeviceMemory d_spheroidal(SIZEOF_SPHEROIDAL);
                cu::DeviceMemory d_aterm(SIZEOF_ATERM);
                cu::DeviceMemory d_grid(SIZEOF_GRID);

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers, SIZEOF_WAVENUMBERS);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal, SIZEOF_SPHEROIDAL);
                htodstream.memcpyHtoDAsync(d_aterm, aterm, SIZEOF_ATERM);
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
                    cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                    cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + s * visibilities_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[5];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, current_jobsize * SIZEOF_VISIBILITIES);
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, current_jobsize * SIZEOF_UVW);
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                            htodstream.record(inputReady);

                            // Create FFT plan
                            kernel_fft->plan(subgridsize, current_jobsize);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            executestream.waitEvent(outputFree);
                            powerRecords[0].enqueue(executestream);
                            kernel_gridder->launch(
                                executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            executestream.record(inputFree);
                            powerRecords[1].enqueue(executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_jobsize, d_subgrids);
                            powerRecords[3].enqueue(executestream);

                            // Launch adder kernel
                            kernel_adder->launch(
                                executestream, current_jobsize,
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
                                                     kernel_gridder->flops(current_jobsize),
                                                     kernel_gridder->bytes(current_jobsize),
                                                     PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("    fft", runtime_fft,
                                                     kernel_fft->flops(subgridsize, current_jobsize),
                                                     kernel_fft->bytes(subgridsize, current_jobsize),
                                                     PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report(" scaler", runtime_scaler,
                                                     kernel_scaler->flops(current_jobsize),
                                                     kernel_scaler->bytes(current_jobsize),
                                                     PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                        auxiliary::report("  adder", runtime_adder,
                                                     kernel_adder->flops(current_jobsize),
                                                     kernel_adder->bytes(current_jobsize),
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
                dtohstream.memcpyDtoHAsync(grid, d_grid, SIZEOF_GRID);
                dtohstream.synchronize();
                executestream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_gridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_subgrids);
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
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
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
                const std::complex<float> *aterm,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Initialize metadata
                vector<Metadata> metadata = init_metadata(uvw, wavenumbers, baselines);
                auto nr_subgrids = metadata.size();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
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
                cu::HostMemory h_visibilities((void *) visibilities, nr_subgrids * SIZEOF_VISIBILITIES);
                cu::HostMemory h_uvw((void *) uvw, nr_subgrids * SIZEOF_UVW);
                cu::HostMemory h_metadata((void *) metadata.data(), nr_subgrids * SIZEOF_METADATA);

                // Device memory
                cu::DeviceMemory d_wavenumbers(SIZEOF_WAVENUMBERS);
                cu::DeviceMemory d_spheroidal(SIZEOF_SPHEROIDAL);
                cu::DeviceMemory d_aterm(SIZEOF_ATERM);
                cu::DeviceMemory d_grid(SIZEOF_GRID);

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers, SIZEOF_WAVENUMBERS);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal, SIZEOF_SPHEROIDAL);
                htodstream.memcpyHtoDAsync(d_aterm, aterm, SIZEOF_ATERM);
                htodstream.memcpyHtoDAsync(d_grid, grid);
                htodstream.synchronize();

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_splitter = 0;
                total_runtime_degridding = -omp_get_wtime();
                PowerSensor::State startState = powerSensor.read();

                // Start gridder
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
                    cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                    cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + s * visibilities_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[5];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, current_jobsize * SIZEOF_VISIBILITIES);
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, current_jobsize * SIZEOF_UVW);
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                            htodstream.record(inputReady);

                            // Create FFT plan
                            kernel_fft->plan(subgridsize, current_jobsize);

                            // Launch splitter kernel
                            executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
                            kernel_splitter->launch(
                                executestream, current_jobsize,
                                d_metadata, d_subgrids, d_grid);
                            powerRecords[1].enqueue(executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_jobsize, d_subgrids);
                            powerRecords[3].enqueue(executestream);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            kernel_degridder->launch(
                                executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            executestream.record(outputReady);
                            powerRecords[4].enqueue(executestream);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
        					dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        double runtime_splitter  = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft       = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler    = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_degridder = PowerSensor::seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter->flops(current_jobsize),
                                                       kernel_splitter->bytes(current_jobsize),
                                                       PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("      fft", runtime_fft,
                                                       kernel_fft->flops(subgridsize, current_jobsize),
                                                       kernel_fft->bytes(subgridsize, current_jobsize),
                                                       PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report("   scaler", runtime_scaler,
                                                       kernel_scaler->flops(current_jobsize),
                                                       kernel_scaler->bytes(current_jobsize),
                                                       PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder->flops(current_jobsize),
                                                       kernel_degridder->bytes(current_jobsize),
                                                       PowerSensor::Watt(powerRecords[3].state, powerRecords[4].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter  += runtime_splitter;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_scaler    += runtime_scaler;
                        total_runtime_degridder += runtime_degridder;
                        #endif
                    } // end for s
                }

                dtohstream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_degridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                uint64_t total_flops_splitter   = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_scaler     = kernel_scaler->flops(nr_subgrids);
                uint64_t total_bytes_scaler     = kernel_scaler->bytes(nr_subgrids);
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_scaler + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_scaler + total_bytes_splitter;
                double   total_watt_degridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
            }


            /* Low level routines */
            void CUDA::transform(DomainAtoDomainB direction, cu::Context &context, cu::HostMemory &h_grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                // Load kernels
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Initialize
                cu::Stream stream;
                context.setCurrent();

                // Performance measurements
                PowerRecord powerRecords[4];

                // Copy grid to device
                cu::DeviceMemory d_grid(SIZEOF_GRID);
                powerRecords[0].enqueue(stream);
                stream.memcpyHtoDAsync(d_grid, h_grid, SIZEOF_GRID);

                // Execute fft
                kernel_fft->plan(gridsize, 1);
                powerRecords[1].enqueue(stream);
                kernel_fft->launch(stream, d_grid, sign);
                powerRecords[2].enqueue(stream);

                // Copy grid to host
                stream.memcpyDtoHAsync(h_grid, d_grid, SIZEOF_GRID);
                powerRecords[3].enqueue(stream);
                stream.synchronize();

                #if defined(REPORT_TOTAL)
                auxiliary::report(" input",
                                  PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state),
                                  0, SIZEOF_GRID,
                                  PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                auxiliary::report("   fft",
                                  PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                auxiliary::report("output",
                                  PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state),
                                  0, SIZEOF_GRID,
                                  PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                std::cout << std::endl;
                #endif
            }


            void CUDA::grid_onto_subgrids(CU_GRIDDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Gridder> kernel_gridder = get_kernel_gridder();
                unique_ptr<Scaler> kernel_scaler = get_kernel_scaler();
                cu::Module *module_fft = (modules[which_module[name_fft]]);

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Performance measurements
                double total_runtime_gridding = 0;
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
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
                    cu::Event outputReady;
                    unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                    // Private device memory
                	cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                	cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + s * visibilities_elements;
                        void *subgrids_ptr     = (complex<float>*) h_subgrids + s * subgrid_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[4];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                		{
                			// Copy input data to device
                			htodstream.waitEvent(inputFree);
                			htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, current_jobsize * SIZEOF_VISIBILITIES);
                			htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, current_jobsize * SIZEOF_UVW);
                			htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                			htodstream.record(inputReady);

                			// Create FFT plan
                            kernel_fft->plan(subgridsize, current_jobsize);

                			// Launch gridder kernel
                			executestream.waitEvent(inputReady);
                			executestream.waitEvent(outputFree);
                            powerRecords[0].enqueue(executestream);

                            kernel_gridder->launch(
                				executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
                				d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[1].enqueue(executestream);

                			// Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_jobsize, d_subgrids);
                            powerRecords[3].enqueue(executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                			// Copy subgrid to host
                			dtohstream.waitEvent(outputReady);
                			dtohstream.memcpyDtoHAsync(subgrids_ptr, d_subgrids, current_jobsize * SIZEOF_SUBGRIDS);
                			dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();

                        double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler  = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder->flops(current_jobsize),
                                                     kernel_gridder->bytes(current_jobsize),
                                                     PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("    fft", runtime_fft,
                                                     kernel_fft->flops(subgridsize, current_jobsize),
                                                     kernel_fft->bytes(subgridsize, current_jobsize),
                                                     PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report(" scaler", runtime_scaler,
                                                     kernel_scaler->flops(current_jobsize),
                                                     kernel_scaler->bytes(current_jobsize),
                                                     PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_gridder += runtime_gridder;
                        total_runtime_fft     += runtime_fft;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_gridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft;
                double   total_watt_gridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
            }


            void CUDA::add_subgrids_to_grid(CU_ADDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Adder> kernel_adder = get_kernel_adder();

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;
                context.setCurrent();

                // Shared device memory
                cu::DeviceMemory d_grid(SIZEOF_GRID);
                htodstream.memcpyHtoDAsync(d_grid, h_grid, SIZEOF_GRID);
                htodstream.synchronize();

                // Performance measurements
                double total_runtime_adding = 0;
                double total_runtime_adder = 0;
                total_runtime_adding = -omp_get_wtime();
                PowerSensor::State startState = powerSensor.read();

                // Start adder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event inputReady;
                    cu::Event executeFinished;

                    // Private device memory
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int subgrid_elements   = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements  = 5;

                        // Pointers to data for current batch
                        void *subgrids_ptr     = (complex<float>*) h_subgrids + s * subgrid_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[2];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, current_jobsize * SIZEOF_SUBGRIDS);
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                            htodstream.record(inputReady);

                            // Launch adder kernel
                            executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
                            kernel_adder->launch(
                                executestream, current_jobsize,
                                d_metadata, d_subgrids, d_grid);
                            powerRecords[1].enqueue(executestream);
                            executestream.record(executeFinished);
                        }

                        executeFinished.synchronize();

                        double runtime_adder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("adder", runtime_adder,
                                                   kernel_adder->flops(current_jobsize),
                                                   kernel_adder->bytes(current_jobsize),
                                                   PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_adder += runtime_adder;
                        #endif
                    } // end for s
                }

                // Copy grid to host
                dtohstream.synchronize();
                dtohstream.memcpyDtoHAsync(h_grid, d_grid, SIZEOF_GRID);
                dtohstream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_adding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                uint64_t total_flops_adder  = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder  = kernel_adder->bytes(nr_subgrids);
                double   total_watt_adding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|adding", total_runtime_adding, total_flops_adder, total_bytes_adder, total_watt_adding);
                auxiliary::report_subgrids("|adding", total_runtime_adding, nr_subgrids);
                clog << endl;
                #endif
            }


            void CUDA::split_grid_into_subgrids(CU_SPLITTER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Splitter> kernel_splitter = get_kernel_splitter();

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;
                context.setCurrent();

                // Shared device memory
                cu::DeviceMemory d_grid(SIZEOF_GRID);
                htodstream.memcpyHtoDAsync(d_grid, h_grid, SIZEOF_GRID);
                htodstream.synchronize();

                // Performance measurements
                double total_runtime_splitting = 0;
                double total_runtime_splitter = 0;
                total_runtime_splitting = -omp_get_wtime();
                PowerSensor::State startState = powerSensor.read();

                // Start adder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event outputReady;
                    cu::Event outputFree;

                    // Private device memory
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int subgrid_elements   = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements  = 5;

                        // Pointers to data for current batch
                        void *subgrids_ptr     = (complex<float>*) h_subgrids + s * subgrid_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[2];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                           // Launch splitter kernel
                            powerRecords[0].enqueue(executestream);
                            kernel_splitter->launch(
                                executestream, current_jobsize,
                                d_metadata, d_subgrids, d_grid);
                            powerRecords[1].enqueue(executestream);
                            executestream.record(outputReady);

                            // Copy subgrid to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, current_jobsize * SIZEOF_SUBGRIDS);
                            dtohstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                            dtohstream.record(outputFree);
                         }

                        outputFree.synchronize();

                        double runtime_splitter = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("splitter", runtime_splitter,
                                                      kernel_splitter->flops(current_jobsize),
                                                      kernel_splitter->bytes(current_jobsize),
                                                      PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter += runtime_splitter;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_splitting += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                uint64_t total_flops_splitter  = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter  = kernel_splitter->bytes(nr_subgrids);
                double   total_watt_splitting  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|splitting", total_runtime_splitting, total_flops_splitter, total_bytes_splitter, total_watt_splitting);
                auxiliary::report_subgrids("|splitting", total_runtime_splitting, nr_subgrids);
                clog << endl;
                #endif
            }


            void CUDA::degrid_from_subgrids(CU_DEGRIDDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size();

                // Load kernel
                unique_ptr<Degridder> kernel_degridder = get_kernel_degridder();
                cu::Module *module_fft = (modules[which_module[name_fft]]);

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
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
                    int current_jobsize = jobsize;
                    unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                	// Private device memory
                    cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                    cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                	cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + s * visibilities_elements;
                        void *subgrids_ptr     = (complex<float>*) h_subgrids + s * subgrid_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[3];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
        				{
        					// Copy input data to device
        					htodstream.waitEvent(inputFree);
        					htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, current_jobsize * SIZEOF_UVW);
        					htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, current_jobsize * SIZEOF_SUBGRIDS);
        					htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
        					htodstream.record(inputReady);

        					// Create FFT plan
                            kernel_fft->plan(subgridsize, current_jobsize);

        					// Launch FFT
        					executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_FORWARD);
                            powerRecords[1].enqueue(executestream);

        					// Launch degridder kernel
        					executestream.waitEvent(outputFree);
                            kernel_degridder->launch(
        						executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
        						d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[2].enqueue(executestream);
        					executestream.record(outputReady);
        					executestream.record(inputFree);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
        					dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
        					dtohstream.record(outputFree);
        				}

        				outputFree.synchronize();
                        double runtime_fft       = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("      fft", PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state),
                                                       kernel_fft->flops(subgridsize, current_jobsize),
                                                       kernel_fft->bytes(subgridsize, current_jobsize),
                                                       PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("degridder", PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state),
                                                       kernel_degridder->flops(current_jobsize),
                                                       kernel_degridder->bytes(current_jobsize),
                                                       PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_degridder += runtime_degridder;
                        total_runtime_fft       += runtime_fft;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_degridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                double   total_watt_degridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
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
                  mParams.get_nr_timesteps(),
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

                string parameters = " " + flags +
                                    " " + compiler_parameters +
                                    " " + mparameters;

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

                    cout << lib << " " << source << " " << endl;

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
