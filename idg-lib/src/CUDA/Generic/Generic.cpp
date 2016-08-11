#include "Generic.h"

#include "../common/CU.h"
#include "../common/DeviceInstance.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            Generic::Generic(
                Parameters params,
                ProxyInfo info) :
                CUDA(params, info)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                // Allocate memory
                for (DeviceInstance *device : devices) {
                    #if REDUCE_HOST_MEMORY
                    h_visibilities_.push_back(new cu::HostMemory(sizeof_visibilities(params.get_nr_baselines())));
                    h_uvw_.push_back(new cu::HostMemory(sizeof_uvw(params.get_nr_baselines())));
                    #endif
                    h_grid_.push_back(new cu::HostMemory(sizeof_grid()));
                }
                #if !REDUCE_HOST_MEMORY
                h_visibilities_ = new cu::HostMemory(sizeof_visibilities(params.get_nr_baselines()));
                h_uvw_ = new cu::HostMemory(sizeof_uvw(params.get_nr_baselines()));
                #endif

                // Setup benchmark
                init_benchmark();
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
                PowerSensor *power_sensor = device->get_powersensor();

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                // Initialize
                cu::Context &context = device->get_context();
                context.setCurrent();

                // Host memory
                #if REDUCE_HOST_MEMORY
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
                PowerRecord powerRecords[5];

                for (int i = 0; i < nr_repetitions; i++) {

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                kernel_fft->shift(h_grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                cu::DeviceMemory d_grid(sizeof_grid());
                device->measure(powerRecords[0], stream);
                stream.memcpyHtoDAsync(d_grid, h_grid, sizeof_grid());
                device->measure(powerRecords[1], stream);

                // Execute fft
                kernel_fft->plan(gridsize, 1);
                device->measure(powerRecords[2], stream);
                kernel_fft->launch(stream, d_grid, sign);
                device->measure(powerRecords[3], stream);

                // Copy grid to host
                stream.memcpyDtoHAsync(h_grid, d_grid, sizeof_grid());
                device->measure(powerRecords[4], stream);
                stream.synchronize();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                kernel_fft->shift(h_grid);
                time_shift += omp_get_wtime();

                // Copy grid from h_grid to grid
                #if !REDUCE_HOST_MEMORY
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
                auxiliary::report("     input",
                                  power_sensor->seconds(powerRecords[0].state, powerRecords[1].state),
                                  0, sizeof_grid(),
                                  power_sensor->Watt(powerRecords[0].state, powerRecords[1].state));
                auxiliary::report("  plan-fft",
                                  power_sensor->seconds(powerRecords[1].state, powerRecords[2].state),
                                  0, 0, 0);
                auxiliary::report("  grid-fft",
                                  power_sensor->seconds(powerRecords[2].state, powerRecords[3].state),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  power_sensor->Watt(powerRecords[2].state, powerRecords[3].state));
                auxiliary::report("    output",
                                  power_sensor->seconds(powerRecords[3].state, powerRecords[4].state),
                                  0, sizeof_grid(),
                                  power_sensor->Watt(powerRecords[3].state, powerRecords[4].state));
                auxiliary::report("  fftshift", time_shift/2, 0, sizeof_grid() * 2, 0);
                if (direction == FourierDomainToImageDomain) {
                auxiliary::report("grid-scale", time_scale/2, 0, sizeof_grid() * 2, 0);
                }
                std::cout << std::endl;
                #endif

                } // end for repetitions
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

                // Configuration
                const int nr_devices = devices.size();
                const int nr_streams = 3;

                // Constants
                auto nr_stations      = mParams.get_nr_stations();
                auto nr_baselines     = mParams.get_nr_baselines();
                auto nr_time          = mParams.get_nr_time();
                auto nr_timeslots     = mParams.get_nr_timeslots();
                auto nr_channels      = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize         = mParams.get_grid_size();
                auto subgridsize      = mParams.get_subgrid_size();
                auto jobsize          = mParams.get_job_size_gridder();
                auto imagesize = mParams.get_imagesize();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Host memory
                #if !REDUCE_HOST_MEMORY
                cu::HostMemory   &h_visibilities = *h_visibilities_;
                cu::HostMemory   &h_uvw          = *h_uvw_;
                h_visibilities.set(visibilities);
                h_uvw.set(uvw);
                #endif

                // Device memory
                std::vector<cu::DeviceMemory*> d_wavenumbers_;
                std::vector<cu::DeviceMemory*> d_spheroidal_;
                std::vector<cu::DeviceMemory*> d_aterm_;
                std::vector<cu::DeviceMemory*> d_grid_;
                for (DeviceInstance *device : devices) {
                    cu::Context &context = device->get_context();
                    context.setCurrent();
                    d_wavenumbers_.push_back(new cu::DeviceMemory(sizeof_wavenumbers()));
                    d_spheroidal_.push_back(new cu::DeviceMemory(sizeof_spheroidal()));
                    d_aterm_.push_back(new cu::DeviceMemory(sizeof_aterm()));
                    d_grid_.push_back(new cu::DeviceMemory(sizeof_grid()));
                }

                // Performance measurements
                double total_runtime_gridder  = 0;
                double total_runtime_fft      = 0;
                double total_runtime_scaler   = 0;
                double total_runtime_adder    = 0;
                double total_runtime_gridding = 0;
                PowerSensor::State startStates[nr_devices];
                PowerSensor::State stopStates[nr_devices];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;

                    // Load device
                    DeviceInstance *device = devices[device_id];
                    PowerSensor *power_sensor = device->get_powersensor();

                    // Load kernels
                    unique_ptr<Gridder> kernel_gridder = device->get_kernel_gridder();
                    unique_ptr<Scaler>  kernel_scaler  = device->get_kernel_scaler();
                    unique_ptr<Adder>   kernel_adder   = device->get_kernel_adder();

                    // Load CUDA objects
                    cu::Context &context      = device->get_context();
                    cu::Stream &executestream = device->get_execute_stream();
                    cu::Stream &htodstream    = device->get_htod_stream();
                    cu::Stream &dtohstream    = device->get_dtoh_stream();
                    context.setCurrent();

                    // Load memory objects
                    #if REDUCE_HOST_MEMORY
                    cu::HostMemory   &h_visibilities = *(h_visibilities_[device_id]);
                    cu::HostMemory   &h_uvw          = *(h_uvw_[device_id]);
                    #endif
                    cu::DeviceMemory &d_wavenumbers  = *(d_wavenumbers_[device_id]);
                    cu::DeviceMemory &d_spheroidal   = *(d_spheroidal_[device_id]);
                    cu::DeviceMemory &d_aterm        = *(d_aterm_[device_id]);
                    cu::DeviceMemory &d_grid         = *(d_grid_[device_id]);
                    cu::HostMemory   &h_grid         = *(h_grid_[device_id]);

                    // Copy read-only device memory
                    if (local_id == 0) {
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                        htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                        htodstream.memcpyHtoDAsync(d_aterm, aterm);
                        d_grid.zero();
                    }
                    htodstream.synchronize();

                    // Initialize
                    cu::Event inputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();

                    // Private device memory
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cu::DeviceMemory d_visibilities(sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(sizeof_uvw(jobsize));
                    cu::DeviceMemory d_subgrids(sizeof_subgrids(max_nr_subgrids));
                    cu::DeviceMemory d_metadata(sizeof_metadata(max_nr_subgrids));

                    // Create FFT plan
                    context.setCurrent();
                    kernel_fft->plan(subgridsize, max_nr_subgrids);

                    // Power measurement
                    PowerRecord powerRecords[5];
                    if (local_id == 0) {
                        startStates[device_id] = device->measure();
                    }

                    for (int i = 0; i < nr_repetitions; i++) {
                    #pragma omp barrier
                    #pragma omp single
                    total_runtime_gridding = -omp_get_wtime();
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in batch
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        #if REDUCE_HOST_MEMORY
                        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        memcpy(h_visibilities, visibilities_ptr, sizeof_visibilities(current_nr_baselines));
                        memcpy(h_uvw, uvw_ptr, sizeof_uvw(current_nr_baselines));
                        uvw_ptr                = h_uvw;
                        visibilities_ptr       = h_visibilities;
                        #else
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        #endif
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerRecord powerRecords[5];

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device memory
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            device->measure(powerRecords[0], executestream);
                            kernel_gridder->launch(
                                executestream, current_nr_subgrids, gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
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
                                executestream, current_nr_subgrids, gridsize,
                                d_metadata, d_subgrids, d_grid);

                            device->measure(powerRecords[4], executestream);
                            executestream.record(outputReady);
                        }

                        outputReady.synchronize();

                        double runtime_gridder = power_sensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = power_sensor->seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler  = power_sensor->seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_adder   = power_sensor->seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                     kernel_gridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                                     power_sensor->Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("sub-fft", runtime_fft,
                                                     kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                     kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                     power_sensor->Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report(" scaler", runtime_scaler,
                                                     kernel_scaler->flops(current_nr_subgrids),
                                                     kernel_scaler->bytes(current_nr_subgrids),
                                                     power_sensor->Watt(powerRecords[2].state, powerRecords[3].state));
                        auxiliary::report("  adder", runtime_adder,
                                                     kernel_adder->flops(current_nr_subgrids),
                                                     kernel_adder->bytes(current_nr_subgrids),
                                                     power_sensor->Watt(powerRecords[3].state, powerRecords[4].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        #pragma omp critical
                        {
                            total_runtime_gridder += runtime_gridder;
                            total_runtime_fft     += runtime_fft;
                            total_runtime_scaler  += runtime_scaler;
                            total_runtime_adder   += runtime_adder;
                        }
                        #endif
                    } // end for bl
                    } // end for repetitions

                    // Wait for all jobs to finish
                    executestream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = device->measure();
                    }

                    // Copy grid to host
                    if (local_id == 0) {
                        dtohstream.memcpyDtoHAsync(h_grid, d_grid, sizeof_grid());
                    }
                    dtohstream.synchronize();
                } // end omp parallel

                total_runtime_gridding += omp_get_wtime();

                // Add new grids to existing grid
                for (int d = 0; d < devices.size(); d++) {
                    float2 *grid_src = (float2 *) *(h_grid_[d]);
                    float2 *grid_dst = (float2 *) grid;

                    #pragma omp parallel for
                    for (int i = 0; i < gridsize * gridsize * nr_polarizations; i++) {
                        grid_dst[i] += grid_src[i];
                    }
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                unique_ptr<GridFFT> kernel_fft     = devices[0]->get_kernel_fft();
                unique_ptr<Gridder> kernel_gridder = devices[0]->get_kernel_gridder();
                unique_ptr<Scaler>  kernel_scaler  = devices[0]->get_kernel_scaler();
                unique_ptr<Adder>   kernel_adder   = devices[0]->get_kernel_adder();
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
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                for (int d = 0; d < devices.size(); d++) {
                    PowerSensor *power_sensor = devices[d]->get_powersensor();
                    double seconds = power_sensor->seconds(startStates[d], stopStates[d]);
                    double watts   = power_sensor->Watt(startStates[d], stopStates[d]);
                    auxiliary::report("|gridding", seconds, 0, 0, watts);
                }
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

                // Configuration
                const int nr_devices = devices.size();
                const int nr_streams = 3;

                // Constants
                auto nr_stations      = mParams.get_nr_stations();
                auto nr_baselines     = mParams.get_nr_baselines();
                auto nr_time          = mParams.get_nr_time();
                auto nr_timeslots     = mParams.get_nr_timeslots();
                auto nr_channels      = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize         = mParams.get_grid_size();
                auto subgridsize      = mParams.get_subgrid_size();
                auto jobsize          = mParams.get_job_size_degridder();
                auto imagesize = mParams.get_imagesize();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Host memory
                #if !REDUCE_HOST_MEMORY
                cu::HostMemory   &h_visibilities = *h_visibilities_;
                cu::HostMemory   &h_uvw          = *h_uvw_;
                h_uvw.set(uvw);
                #endif
                cu::HostMemory &h_grid = *(h_grid_[0]);
                h_grid.set(grid);

                // Device memory
                std::vector<cu::DeviceMemory*> d_wavenumbers_;
                std::vector<cu::DeviceMemory*> d_spheroidal_;
                std::vector<cu::DeviceMemory*> d_aterm_;
                std::vector<cu::DeviceMemory*> d_grid_;
                for (DeviceInstance *device : devices) {
                    cu::Context &context = device->get_context();
                    context.setCurrent();
                    d_wavenumbers_.push_back(new cu::DeviceMemory(sizeof_wavenumbers()));
                    d_spheroidal_.push_back(new cu::DeviceMemory(sizeof_spheroidal()));
                    d_aterm_.push_back(new cu::DeviceMemory(sizeof_aterm()));
                    d_grid_.push_back(new cu::DeviceMemory(sizeof_grid()));
                }

                // Performance measurements
                double total_runtime_degridder  = 0;
                double total_runtime_fft        = 0;
                double total_runtime_splitter   = 0;
                double total_runtime_degridding = 0;
                PowerSensor::State startStates[nr_devices];
                PowerSensor::State stopStates[nr_devices];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id = global_id % nr_streams;

                    // Load device
                    DeviceInstance *device = devices[device_id];
                    PowerSensor *power_sensor = device->get_powersensor();

                    // Load kernels
                    unique_ptr<Degridder> kernel_degridder = device->get_kernel_degridder();
                    unique_ptr<Splitter>  kernel_splitter  = device->get_kernel_splitter();
                    unique_ptr<GridFFT>   kernel_fft       = device->get_kernel_fft();

                    // Load CUDA objects
                    cu::Context &context      = device->get_context();
                    cu::Stream &executestream = device->get_execute_stream();
                    cu::Stream &htodstream    = device->get_htod_stream();
                    cu::Stream &dtohstream    = device->get_dtoh_stream();
                    context.setCurrent();

                    // Load memory objects
                    #if REDUCE_HOST_MEMORY
                    cu::HostMemory   &h_visibilities = *(h_visibilities_[device_id]);
                    cu::HostMemory   &h_uvw          = *(h_uvw_[device_id]);
                    #endif
                    cu::DeviceMemory &d_wavenumbers  = *(d_wavenumbers_[device_id]);
                    cu::DeviceMemory &d_spheroidal   = *(d_spheroidal_[device_id]);
                    cu::DeviceMemory &d_aterm        = *(d_aterm_[device_id]);
                    cu::DeviceMemory &d_grid         = *(d_grid_[device_id]);

                    // Copy read-only device memory
                    if (local_id == 0) {
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                        htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                        htodstream.memcpyHtoDAsync(d_aterm, aterm);
                        htodstream.memcpyHtoDAsync(d_grid, h_grid);
                    }
                    htodstream.synchronize();

                    // Events
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;

                    // Private device memory
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cu::DeviceMemory d_visibilities(sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(sizeof_uvw(jobsize));
                    cu::DeviceMemory d_subgrids(sizeof_subgrids(max_nr_subgrids));
                    cu::DeviceMemory d_metadata(sizeof_metadata(max_nr_subgrids));

                    // Create FFT plan
                    context.setCurrent();
                    kernel_fft->plan(subgridsize, max_nr_subgrids);

                    // Power measurement
                    PowerRecord powerRecords[5];
                    if (local_id == 0) {
                        startStates[device_id] = device->measure();
                    }

                    for (int i = 0; i < nr_repetitions; i++) {
                    #pragma omp barrier
                    #pragma omp single
                    total_runtime_degridding = -omp_get_wtime();
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in batch
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        #if REDUCE_HOST_MEMORY
                        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        memcpy(h_uvw, uvw_ptr, sizeof_uvw(current_nr_baselines));
                        uvw_ptr                = h_uvw;
                        visibilities_ptr       = h_visibilities;
                        #else
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        #endif
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerRecord powerRecords[5];

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch splitter kernel
                            executestream.waitEvent(inputReady);
                            device->measure(powerRecords[0], executestream);
                            kernel_splitter->launch(
                                executestream, current_nr_subgrids, gridsize,
                                d_metadata, d_subgrids, d_grid);
                            device->measure(powerRecords[1], executestream);

                            // Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_FORWARD);
                            device->measure(powerRecords[2], executestream);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device->measure(powerRecords[3], executestream);
                            kernel_degridder->launch(
                                executestream, current_nr_subgrids, gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            device->measure(powerRecords[4], executestream);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, sizeof_visibilities(current_nr_baselines));
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        #if REDUCE_HOST_MEMORY
                        visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        memcpy(visibilities_ptr, h_visibilities, sizeof_visibilities(current_nr_baselines));
                        #endif

                        double runtime_splitter  = power_sensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft       = power_sensor->seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_degridder = power_sensor->seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter->flops(current_nr_subgrids),
                                                       kernel_splitter->bytes(current_nr_subgrids),
                                                       power_sensor->Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("  sub-fft", runtime_fft,
                                                       kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                       kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                       power_sensor->Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                       kernel_degridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                                       power_sensor->Watt(powerRecords[3].state, powerRecords[4].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter  += runtime_splitter;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_degridder += runtime_degridder;
                        #endif
                    } // end for bl
                    } // end for repetitions

                    // Wait for all jobs to finish
                    dtohstream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = device->measure();
                    }
                } // end omp parallel

                total_runtime_degridding += omp_get_wtime();

                // Copy visibilities from cuda h_visibilities to visibilities
                #if !REDUCE_HOST_MEMORY
                memcpy(visibilities, h_visibilities, sizeof_visibilities(nr_baselines));
                #endif

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                unique_ptr<Splitter>  kernel_splitter  = devices[0]->get_kernel_splitter();
                unique_ptr<GridFFT>   kernel_fft       = devices[0]->get_kernel_fft();
                unique_ptr<Degridder> kernel_degridder = devices[0]->get_kernel_degridder();
                uint64_t total_flops_splitter   = kernel_splitter->flops(total_nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(total_nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, total_nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, total_nr_subgrids);
                uint64_t total_flops_degridder  = kernel_degridder->flops(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                for (int d = 0; d < devices.size(); d++) {
                    PowerSensor *power_sensor = devices[d]->get_powersensor();
                    double seconds = power_sensor->seconds(startStates[d], stopStates[d]);
                    double watts   = power_sensor->Watt(startStates[d], stopStates[d]);
                    auxiliary::report("|degridding", seconds, 0, 0, watts);
                }
                clog << endl;
                #endif
            }

            void Generic::init_benchmark() {
                char *char_nr_repetitions = getenv("NR_REPETITIONS");
                if (char_nr_repetitions) {
                    nr_repetitions = atoi(char_nr_repetitions);
                    enable_benchmark = nr_repetitions > 1;
                }
                if (enable_benchmark) {
                    std::clog << "Benchmark mode enabled, nr_repetitions = " << nr_repetitions << std::endl;
                }
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
