#include "Generic.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            Generic::Generic(
                Parameters params,
                ProxyInfo info) :
                info(info)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                cout << params;
                #endif

                mParams = params;
                cu::init();
                init_devices();
                print_devices();
                print_compiler_flags();
            }

            void Generic::init_devices() {
                // The list of CUDA devices to use are read from the environment
                char *char_cuda_device = getenv("CUDA_DEVICE");

                // Get list of all device numbers
                vector<int> device_numbers;

                if (!char_cuda_device) {
                    // Use device 0 if no CUDA devices were specified
                    device_numbers.push_back(0);
                } else if (strlen(char_cuda_device) == 1) {
                    // Just one device number was specified
                    device_numbers.push_back(atoi(char_cuda_device));
                } else {
                    // Split device numbers on comma
                    const char *delimiter = (char *) ",";
                    char *token = strtok(char_cuda_device, delimiter);
                    if (token) device_numbers.push_back(atoi(token));
                    while (token) {
                        token = strtok(NULL, delimiter);
                        if (token) device_numbers.push_back(atoi(token));
                    }
                }

                // Create a device instance for every device
                for (int device_number : device_numbers) {
                    DeviceInstance *device = new DeviceInstance(mParams, info, device_number);
                    devices.push_back(device);
                }
            }

            void Generic::print_devices() {
                std::cout << "Devices: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << *device;
                }
                std::cout << std::endl;
            }

            void Generic::print_compiler_flags() {
                std::cout << "Compiler flags: " << std::endl;
                for (DeviceInstance *device : devices) {
                    std::cout << device->get_compiler_flags() << std::endl;
                }
                std::cout << std::endl;
            }

            std::vector<DeviceInstance*> Generic::get_devices() {
                return devices;
            }

            ProxyInfo Generic::default_info() {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                string srcdir = string(IDG_INSTALL_DIR)
                    + "/lib/kernels/CUDA/";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                char _tmpdir[] = "/tmp/idg-XXXXXX";
                char *tmpdir = mkdtemp(_tmpdir);
                #if defined(DEBUG)
                cout << "Temporary files will be stored in: " << tmpdir << endl;
                #endif

                // Create proxy info
                ProxyInfo p;
                p.set_path_to_src(srcdir);
                p.set_path_to_lib(tmpdir);

                string libgridder = "Gridder.ptx";
                string libdegridder = "Degridder.ptx";
                string libfft = "FFT.ptx";
                string libscaler = "Scaler.ptx";
                string libadder = "Adder.ptx";
                string libsplitter = "Splitter.ptx";

                p.add_lib(libgridder);
                p.add_lib(libdegridder);
                p.add_lib(libfft);
                p.add_lib(libscaler);
                p.add_lib(libadder);
                p.add_lib(libsplitter);

                p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
                p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
                p.add_src_file_to_lib(libfft, "KernelFFT.cu");
                p.add_src_file_to_lib(libscaler, "KernelScaler.cu");
                p.add_src_file_to_lib(libadder, "KernelAdder.cu");
                p.add_src_file_to_lib(libsplitter, "KernelSplitter.cu");

                p.set_delete_shared_objects(true);

                return p;
            }

            /* Sizeof routines */
            uint64_t Generic::sizeof_subgrids(int nr_subgrids) {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t Generic::sizeof_uvw(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                return 1ULL * nr_baselines * nr_time * sizeof(UVW);
            }

            uint64_t Generic::sizeof_visibilities(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                return 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(complex<float>);
            }

            uint64_t Generic::sizeof_metadata(int nr_subgrids) {
                return 1ULL * nr_subgrids * sizeof(Metadata);
            }

            uint64_t Generic::sizeof_grid() {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                return 1ULL * nr_polarizations * gridsize * gridsize * sizeof(complex<float>);
            }

            uint64_t Generic::sizeof_wavenumbers() {
                auto nr_channels = mParams.get_nr_channels();
                return 1ULL * nr_channels * sizeof(float);
            }

            uint64_t Generic::sizeof_aterm() {
                auto nr_stations = mParams.get_nr_stations();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t Generic::sizeof_spheroidal() {
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * subgridsize * subgridsize * sizeof(float);
            }

            /* High level routines */
            void Generic::transform(
                DomainAtoDomainB direction,
                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Load device
                DeviceInstance *device = devices[0];

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                // Initialize
                cu::Context &context = device->get_context();
                context.setCurrent();

                // Host memory
                #if REUSE_HOST_MEMORY
                cu::HostMemory h_grid(grid, sizeof_grid());
                #else
                cu::HostMemory h_grid(sizeof_grid());
                h_grid.set(grid);
                #endif

                // Load kernels
                unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();

                // Initialize
                cu::Stream stream;
                context.setCurrent();

                // Performance measurements
                PowerRecord powerRecords[4];

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                kernel_fft->shift(h_grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                cu::DeviceMemory d_grid(sizeof_grid());
                //powerRecords[0].enqueue(stream);
                device->measure(powerRecords[0], stream);
                stream.memcpyHtoDAsync(d_grid, h_grid, sizeof_grid());

                // Execute fft
                kernel_fft->plan(gridsize, 1);
                device->measure(powerRecords[1], stream);
                kernel_fft->launch(stream, d_grid, sign);
                device->measure(powerRecords[2], stream);

                // Copy grid to host
                stream.memcpyDtoHAsync(h_grid, d_grid, sizeof_grid());
                device->measure(powerRecords[3], stream);
                stream.synchronize();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                kernel_fft->shift(h_grid);
                time_shift += omp_get_wtime();

                // Copy grid from h_grid to grid
                #if !REUSE_HOST_MEMORY
                memcpy(grid, h_grid, sizeof_grid());
                #endif

                // Perform fft scaling
                double time_scale = -omp_get_wtime();
                complex<float> scale = complex<float>(2.0/(gridsize*gridsize), 0);
                if (direction == FourierDomainToImageDomain) {
                    kernel_fft->scale(grid, scale);
                }
                time_scale += omp_get_wtime();


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
                auxiliary::report("fftshift", time_shift/2, 0, sizeof_grid() * 2, 0);
                if (direction == FourierDomainToImageDomain) {
                    auxiliary::report(" scaling", time_scale/2, 0, sizeof_grid() * 2, 0);
                }
                std::cout << std::endl;
                #endif
            }

            void Generic::grid_visibilities(
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

                // Load device
                DeviceInstance *device = devices[0];

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
                unique_ptr<Gridder> kernel_gridder = device->get_kernel_gridder();
                unique_ptr<Scaler> kernel_scaler   = device->get_kernel_scaler();
                unique_ptr<Adder> kernel_adder     = device->get_kernel_adder();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cu::Context &context = device->get_context();
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Host memory
                #if REUSE_HOST_MEMORY
                cu::HostMemory h_visibilities((void *) visibilities, sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw((void *) uvw, sizeof_uvw(nr_baselines));
                #else
                cu::HostMemory h_visibilities(sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw(sizeof_uvw(nr_baselines));
                h_visibilities.set((void *) visibilities);
                h_uvw.set((void *) uvw);
                #endif

                // Device memory
                cu::DeviceMemory d_wavenumbers(sizeof_wavenumbers());
                cu::DeviceMemory d_spheroidal(sizeof_spheroidal());
                cu::DeviceMemory d_aterm(sizeof_aterm());
                cu::DeviceMemory d_grid(sizeof_grid());
                d_grid.zero();

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;
                PowerSensor::State startState;

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                htodstream.memcpyHtoDAsync(d_aterm, aterm);
                htodstream.synchronize();

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();

                    // Private device memory
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cu::DeviceMemory d_visibilities(sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(sizeof_uvw(jobsize));
                    cu::DeviceMemory d_subgrids(sizeof_subgrids(max_nr_subgrids));
                    cu::DeviceMemory d_metadata(sizeof_metadata(max_nr_subgrids));

                    // Power measurement
                    PowerRecord powerRecords[5];
                    #pragma omp single
                    startState = device->measure();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        cuda::PowerRecord powerRecords[5];

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            executestream.waitEvent(outputFree);
                            device->measure(powerRecords[0], executestream);
                            kernel_gridder->launch(
                                executestream, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            device->measure(powerRecords[1], executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            device->measure(powerRecords[2], executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_nr_subgrids, d_subgrids);
                            device->measure(powerRecords[3], executestream);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

                            // Launch adder kernel
                            kernel_adder->launch(
                                executestream, current_nr_subgrids,
                                d_metadata, d_subgrids, d_grid);

                            device->measure(powerRecords[4], executestream);
                            executestream.record(outputReady);
                        }

                        outputReady.synchronize();

                        double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler  = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_adder   = PowerSensor::seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                     kernel_gridder->bytes(current_nr_timesteps, current_nr_subgrids),
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

                // Wait for all jobs to finish
                executestream.synchronize();
                PowerSensor::State stopState = device->measure();

                // Copy grid to host
                dtohstream.memcpyDtoHAsync(grid, d_grid, sizeof_grid());
                dtohstream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();
                uint64_t total_flops_gridder  = kernel_gridder->flops(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, total_nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, total_nr_subgrids);
                uint64_t total_flops_scaler   = kernel_scaler->flops(total_nr_subgrids);
                uint64_t total_bytes_scaler   = kernel_scaler->bytes(total_nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder->flops(total_nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder->bytes(total_nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft + total_flops_scaler + total_flops_adder;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft + total_bytes_scaler + total_bytes_adder;
                double total_runtime_gridding = PowerSensor::seconds(startState, stopState);
                double total_watt_gridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }


            void Generic::degrid_visibilities(
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

                // Load device
                DeviceInstance *device = devices[0];

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
                unique_ptr<Degridder> kernel_degridder = device->get_kernel_degridder();
                unique_ptr<Splitter> kernel_splitter   = device->get_kernel_splitter();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cu::Context &context = device->get_context();
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Host memory
                #if REUSE_HOST_MEMORY
                cu::HostMemory h_visibilities((void *) visibilities, sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw((void *) uvw, sizeof_uvw(nr_baselines));
                #else
                cu::HostMemory h_visibilities(sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw(sizeof_uvw(nr_baselines));
                h_visibilities.zero();
                h_uvw.set((void *) uvw);
                #endif

                // Device memory
                cu::DeviceMemory d_wavenumbers(sizeof_wavenumbers());
                cu::DeviceMemory d_spheroidal(sizeof_spheroidal());
                cu::DeviceMemory d_aterm(sizeof_aterm());
                cu::DeviceMemory d_grid(sizeof_grid());

                // Performance measurements
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_splitter = 0;
                PowerSensor::State startState;

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                htodstream.memcpyHtoDAsync(d_aterm, aterm);
                htodstream.memcpyHtoDAsync(d_grid, grid);
                htodstream.synchronize();

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();

                    // Private device memory
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cu::DeviceMemory d_visibilities(sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(sizeof_uvw(jobsize));
                    cu::DeviceMemory d_subgrids(sizeof_subgrids(max_nr_subgrids));
                    cu::DeviceMemory d_metadata(sizeof_metadata(max_nr_subgrids));

                    // Power measurement
                    PowerRecord powerRecords[5];
                    #pragma omp single
                    startState = device->measure();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in job
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);

                        // Pointers to data for current job
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        void *metadata_ptr     = (int *) plan.get_metadata_ptr(bl);

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                            // Launch splitter kernel
                            executestream.waitEvent(inputReady);
                            device->measure(powerRecords[0], executestream);
                            kernel_splitter->launch(
                                executestream, current_nr_subgrids,
                                d_metadata, d_subgrids, d_grid);
                            device->measure(powerRecords[1], executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_FORWARD);
                            device->measure(powerRecords[2], executestream);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device->measure(powerRecords[3], executestream);
                            kernel_degridder->launch(
                                executestream, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            device->measure(powerRecords[4], executestream);
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
                                                       kernel_degridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                       kernel_degridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                                       PowerSensor::Watt(powerRecords[3].state, powerRecords[4].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter  += runtime_splitter;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_degridder += runtime_degridder;
                        #endif
                    } // end for s
                }

                // Wait for all jobs to finish
                dtohstream.synchronize();
                PowerSensor::State stopState = device->measure();

                // Copy visibilities from cuda h_visibilities to visibilities
                #if !REUSE_HOST_MEMORY
                memcpy(visibilities, h_visibilities, sizeof_visibilities(nr_baselines));
                #endif

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();
                uint64_t total_flops_splitter   = kernel_splitter->flops(total_nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(total_nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, total_nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, total_nr_subgrids);
                uint64_t total_flops_degridder  = kernel_degridder->flops(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                double total_runtime_degridding = PowerSensor::seconds(startState, stopState);
                double total_watt_degridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }

        } // namespace cuda
    } // namespace proxy
} // namespace idg


// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::cuda::Generic CUDA_Generic;

    CUDA_Generic* CUDA_Generic_init(
                unsigned int nr_stations,
                unsigned int nr_channels,
                unsigned int nr_time,
                unsigned int nr_timeslots,
                float        imagesize,
                unsigned int grid_size,
                unsigned int subgrid_size)
    {
        idg::Parameters P;
        P.set_nr_stations(nr_stations);
        P.set_nr_channels(nr_channels);
        P.set_nr_time(nr_time);
        P.set_nr_timeslots(nr_timeslots);
        P.set_imagesize(imagesize);
        P.set_subgrid_size(subgrid_size);
        P.set_grid_size(grid_size);

        return new CUDA_Generic(P);
    }

    void CUDA_Generic_grid(CUDA_Generic* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->grid_visibilities(
                (const std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) metadata,
                (std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void CUDA_Generic_degrid(CUDA_Generic* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->degrid_visibilities(
                (std::complex<float>*) visibilities,
                    (const float*) uvw,
                    (const float*) wavenumbers,
                    (const int*) metadata,
                    (const std::complex<float>*) grid,
                    w_offset,
                    kernel_size,
                    (const std::complex<float>*) aterm,
                    (const int*) aterm_offsets,
                    (const float*) spheroidal);
     }

    void CUDA_Generic_transform(CUDA_Generic* p,
                    int direction,
                    void *grid)
    {
       if (direction!=0)
           p->transform(idg::ImageDomainToFourierDomain,
                    (std::complex<float>*) grid);
       else
           p->transform(idg::FourierDomainToImageDomain,
                    (std::complex<float>*) grid);
    }

    void CUDA_Generic_destroy(CUDA_Generic* p) {
       delete p;
    }

} // end extern "C"
