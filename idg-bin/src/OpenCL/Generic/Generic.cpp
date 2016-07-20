#include "Generic.h"

/*
    Toggle between two modes of cu::HostMemory allocation
        REDUCE_HOST_MEMORY = 0:
            visibilities and uvw will be completely mapped
            into host memory shared by all threads
            (this takes some time, especially for large buffers)
        REDUCE_HOST_MEMORY = 1:
            every thread allocates private host memory
            to hold data for just one job
            (throughput is lower, due to additional memory copies)
*/
#define REDUCE_HOST_MEMORY 0

/*
    Toggle warmup
        Copy some memory to device and execute an FFT
        prior to starting the actual computation
*/
#define ENABLE_WARMUP 1

/*
    Toggle planning and execution of Fourier transformations on and off
        The clFFT library contains memory leaks, which makes it much harder
        to find and resolve issues in non-library code. This option disables
        usage of the library so that they can be resolved
*/
#define ENABLE_FFT 1

/*
    When a large amount of data is copies using enqueueWriteBuffer
    the timeline in CodeXL is broken. As a workaround, data is copied
    in many smaller pieces. (Which seems to be faster anyway)
*/
#define PREVENT_CODEXL_BUG 1


using namespace std;
using namespace idg::kernel::opencl;


namespace idg {
    namespace proxy {
        namespace opencl {
            Generic::Generic(
                Parameters params) :
                OpenCL(params)
                #if !REDUCE_HOST_MEMORY
                ,
                h_visibilities(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(params.get_nr_baselines())),
                h_uvw(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(params.get_nr_baselines()))
                #endif
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                // Allocate memory
                for (DeviceInstance *device : devices) {
                    #if REDUCE_HOST_MEMORY
                    h_visibilities_.push_back(new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(params.get_nr_baselines())));
                    h_uvw_.push_back(new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(params.get_nr_baselines())));
                    #endif
                    h_grid_.push_back(new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_grid()));
                }
            }

            Generic::~Generic() {
                for (int i = 0; i < devices.size(); i++) {
                    #if REDUCE_HOST_MEMORY
                    delete h_visibilities_[i];
                    delete h_uvw_[i];
                    #endif
                    delete h_grid_[i];
                }
            }


            /* High level routines */
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
                DeviceInstance *device    = devices[0];
                PowerSensor *power_sensor = device->get_powersensor();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto imagesize = mParams.get_imagesize();
                auto jobsize = mParams.get_job_size_gridder();
                jobsize = nr_baselines < jobsize ? nr_baselines : jobsize;

                // Load kernels
                unique_ptr<Gridder> kernel_gridder = device->get_kernel_gridder();
                unique_ptr<Adder> kernel_adder     = device->get_kernel_adder();
                unique_ptr<Scaler> kernel_scaler   = device->get_kernel_scaler();
                unique_ptr<GridFFT> kernel_fft     = device->get_kernel_fft();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cl::CommandQueue &executequeue = device->get_execute_queue();
                cl::CommandQueue &htodqueue    = device->get_htod_queue();
                cl::CommandQueue &dtohqueue    = device->get_dtoh_queue();
                const int nr_streams = 3;

                // Host memory
                cl::Buffer h_visibilities(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_metadata(total_nr_subgrids));

                // Copy input data to host memory
                #if PREVENT_CODEXL_BUG
                for (int bl = 0; bl < nr_baselines; bl++) {
                    auto offset = bl * sizeof_visibilities(1);
                    const void *visibilities_ptr = visibilities + (offset/sizeof(complex<float>));
                    htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, offset, sizeof_visibilities(1), visibilities_ptr);
                }
                #else
                htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0, sizeof_visibilities(nr_baselines), visibilities);
                #endif
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0, sizeof_metadata(total_nr_subgrids), metadata);

                // Device memory
                cl::Buffer d_wavenumbers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_wavenumbers());
                cl::Buffer d_aterm       = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_aterm());
                cl::Buffer d_spheroidal  = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_spheroidal());
                cl::Buffer d_grid        = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_grid());

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;
                PowerSensor::State startState;

                // Copy static device memory
                htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, sizeof_wavenumbers(), wavenumbers);
                htodqueue.enqueueWriteBuffer(d_aterm, CL_FALSE, 0, sizeof_aterm(), aterm);
                htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, sizeof_spheroidal(), spheroidal);
                htodqueue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid(), grid);

                // Initialize fft
                auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                #if ENABLE_FFT
                kernel_fft->plan(*context, executequeue, subgridsize, max_nr_subgrids);
                #endif

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Private device memory
                    cl::Buffer d_visibilities = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_visibilities(jobsize));
                    cl::Buffer d_uvw          = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids     = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata     = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_metadata(max_nr_subgrids));

                    // Warmup
                    #if ENABLE_WARMUP
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, 0, 0, sizeof_metadata(max_nr_subgrids), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, 0, 0, sizeof_visibilities(jobsize), NULL, NULL);
                    htodqueue.finish();
                    #if ENABLE_FFT
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    #endif
                    executequeue.finish();
                    #endif

                    // Events
                    vector<cl::Event> inputReady(1), outputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);

                    // Performance counters
                    vector<PerformanceCounter> counters(4);
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(power_sensor);
                    }
                    #pragma omp single
                    startState = power_sensor->read();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);
                        auto subgrid_offset       = plan.get_subgrid_offset(bl);

                        // Offsets
                        size_t uvw_offset          = bl * sizeof_uvw(1);
                        size_t visibilities_offset = bl * sizeof_visibilities(1);
                        size_t metadata_offset     = subgrid_offset * sizeof_metadata(1);

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&outputReady, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, sizeof_uvw(current_nr_baselines), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, sizeof_visibilities(current_nr_baselines), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, sizeof_metadata(current_nr_subgrids), NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

							// Launch gridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            kernel_gridder->launchAsync(
                                executequeue, current_nr_timesteps, current_nr_subgrids,
                                gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);

							// Launch FFT
                            #if ENABLE_FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                            #endif

                            // Launch adder kernel
                            kernel_adder->launchAsync(
                                executequeue, current_nr_subgrids, gridsize,
                                d_metadata, d_subgrids, d_grid, counters[3]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
                        }
                    }

                    outputReady[0].wait();
                }

                // Copy grid to host
                executequeue.finish();
                PowerSensor::State stopState = power_sensor->read();
                dtohqueue.enqueueReadBuffer(d_grid, CL_TRUE, 0, sizeof_grid(), grid, NULL, NULL);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
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
                double total_runtime_gridding = power_sensor->seconds(startState, stopState);
                double total_watt_gridding    = power_sensor->Watt(startState, stopState);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            } // end grid_visibilities


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
                auto imagesize        = mParams.get_imagesize();
                auto jobsize          = mParams.get_job_size_degridder();
                jobsize = nr_baselines < jobsize ? nr_baselines : jobsize;

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Host memory
                cl::Buffer &h_grid = *(h_grid_[0]);
                cl::CommandQueue &queue = devices[0]->get_htod_queue();
                queue.enqueueWriteBuffer(h_grid, CL_FALSE, 0, sizeof_grid(), grid);
                #if !REDUCE_HOST_MEMORY
                queue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw(nr_baselines), uvw);
                #endif

                // Device memory
                std::vector<cl::Buffer*> d_wavenumbers_;
                std::vector<cl::Buffer*> d_spheroidal_;
                std::vector<cl::Buffer*> d_aterm_;
                std::vector<cl::Buffer*> d_grid_;
                for (DeviceInstance *device : devices) {
                    d_wavenumbers_.push_back(new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_wavenumbers()));
                    d_spheroidal_.push_back(new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_spheroidal()));
                    d_aterm_.push_back(new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_aterm()));
                    d_grid_.push_back(new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_grid()));
                }

                // Locks
                int locks[nr_devices];

                // Performance measurements
                double total_runtime_degridding = 0;
                double time_degridding_start = 0;
                PowerSensor::State startStates[nr_devices];
                PowerSensor::State stopStates[nr_devices];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id = global_id % nr_streams;

                    // Load device
                    DeviceInstance *device    = devices[device_id];
                    PowerSensor *power_sensor = device->get_powersensor();

                    // Load kernels
                    unique_ptr<Degridder> kernel_degridder = device->get_kernel_degridder();
                    unique_ptr<Splitter>  kernel_splitter  = device->get_kernel_splitter();;
                    unique_ptr<GridFFT>   kernel_fft       = device->get_kernel_fft();

                    // Load OpenCL objects
                    cl::CommandQueue executequeue = device->get_execute_queue();
                    cl::CommandQueue htodqueue    = device->get_htod_queue();
                    cl::CommandQueue dtohqueue    = device->get_dtoh_queue();

                    // Load memory objects
                    #if REDUCE_HOST_MEMORY
                    cl::Buffer &h_visibilities = *(h_visibilities_[device_id]);
                    cl::Buffer &h_uvw          = *(h_uvw_[device_id]);
                    #endif
                    cl::Buffer &d_wavenumbers  = *(d_wavenumbers_[device_id]);
                    cl::Buffer &d_spheroidal   = *(d_spheroidal_[device_id]);
                    cl::Buffer &d_aterm        = *(d_aterm_[device_id]);
                    cl::Buffer &d_grid         = *(d_grid_[device_id]);

                    // Copy read-only device memory
                    if (local_id == 0) {
                        htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, sizeof_wavenumbers(), wavenumbers);
                        htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, sizeof_spheroidal(), spheroidal);
                        htodqueue.enqueueWriteBuffer(d_aterm, CL_FALSE, 0, sizeof_aterm(), aterm);
                        htodqueue.enqueueCopyBuffer(h_grid, d_grid, 0 , 0, sizeof_grid());
                    }
                    htodqueue.finish();

                    // Events
                    vector<cl::Event> inputFree(1), outputFree(1), inputReady(1), outputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &inputFree[0]);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);

                    // Private device memory
                    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cl::Buffer d_visibilities = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_visibilities(jobsize));
                    cl::Buffer d_uvw          = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids     = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata     = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_metadata(max_nr_subgrids));

                    // Create FFT plan
                    #if ENABLE_FFT
                    kernel_fft->plan(*context, executequeue, subgridsize, max_nr_subgrids);
                    #endif

                    // Lock
                    int lock = locks[device_id];

                    // Warmup
                    #if ENABLE_WARMUP
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize), NULL, NULL);
                    htodqueue.finish();
                    #if ENABLE_FFT
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    executequeue.finish();
                    #endif
                    #endif

                    // Performance measurement
                    vector<PerformanceCounter> counters(3);
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(power_sensor);
                    }
                    if (local_id == 0) {
                        startStates[device_id] = power_sensor->read();
                    }

                    #pragma omp barrier
                    #pragma omp single
                    time_degridding_start = omp_get_wtime();
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);
                        auto subgrid_offset       = plan.get_subgrid_offset(bl);

                        // Offsets
                        size_t uvw_offset          = bl * sizeof_uvw(1);
                        size_t visibilities_offset = bl * sizeof_visibilities(1);
                        size_t metadata_offset     = subgrid_offset * sizeof_metadata(1);

                        // Pointers to data for current batch
                        #if REDUCE_HOST_MEMORY
                        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
                        htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw(current_nr_baselines), uvw_ptr);
                        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        uvw_offset = 0;
                        visibilities_offset = 0;
                        #endif
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&inputFree, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, sizeof_uvw(current_nr_baselines));
                            htodqueue.enqueueWriteBuffer(d_metadata, CL_FALSE, 0, sizeof_metadata(current_nr_subgrids), metadata_ptr);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch splitter kernel
                            executequeue.enqueueBarrierWithWaitList(&inputReady, NULL);
                            kernel_splitter->launchAsync(
                                executequeue, current_nr_subgrids, gridsize,
                                d_metadata, d_subgrids, d_grid, counters[0]);

                            // Launch FFT
                            #if ENABLE_FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_FORWARD);
                            #endif

                            // Launch degridder kernel
                            executequeue.enqueueBarrierWithWaitList(&outputFree, NULL);
                            kernel_degridder->launchAsync(
                                executequeue, current_nr_timesteps, current_nr_subgrids,
                                gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[2]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &inputFree[0]);

                            // Copy visibilities to host
                            dtohqueue.enqueueBarrierWithWaitList(&outputReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, sizeof_visibilities(current_nr_baselines), NULL, &outputFree[0]);
                        }

                        #if REDUCE_HOST_MEMORY
                        outputFree[0].wait();
                        dtohqueue.enqueueReadBuffer(h_visibilities, CL_TRUE, 0, sizeof_visibilities(current_nr_baselines), visibilities_ptr);
                        #endif
                    } // end for bl

                    // End degridding timing
                    #pragma atomic
                    {
                        total_runtime_degridding = omp_get_wtime() - time_degridding_start;
                    }

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = power_sensor->read();
                    }

                    // Wait for all jobs to finish
                    dtohqueue.finish();
                } // end omp parallel

                // Free device memory
                for (int d = 0; d < devices.size(); d++) {
                    delete d_wavenumbers_[d];
                    delete d_spheroidal_[d];
                    delete d_aterm_[d];
                    delete d_grid_[d];
                }

                // Copy visibilities from opencl h_visibilities to visibilities
                #if !REDUCE_HOST_MEMORY
                #if PREVENT_CODEXL_BUG
                for (int bl = 0; bl < nr_baselines; bl++) {
                    auto offset = bl * sizeof_visibilities(1);
                    void *visibilities_ptr = visibilities + (offset/sizeof(complex<float>));
                    queue.enqueueReadBuffer(h_visibilities, CL_FALSE, offset, sizeof_visibilities(1), visibilities_ptr);
                }
                queue.finish(); // This line triggers a redundant synchronization warning in CodeXL
                #else
                queue.enqueueReadBuffer(h_visibilities, CL_TRUE, 0, sizeof_visibilities(nr_baselines), visibilities);
                #endif
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
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                for (int d = 0; d < devices.size(); d++) {
                    PowerSensor *power_sensor = devices[d]->get_powersensor();
                    double seconds = power_sensor->seconds(startStates[d], stopStates[d]);
                    double watts   = power_sensor->Watt(startStates[d], stopStates[d]);
                    auxiliary::report("|degridding", seconds, 0, 0, watts);
                }
                clog << endl;
                #endif
            } // end degrid_visibilities

            void Generic::transform(
                DomainAtoDomainB direction,
                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Load device
                DeviceInstance *device    = devices[0];
                PowerSensor *power_sensor = device->get_powersensor();

                // Constants
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                clfftDirection sign = (direction == FourierDomainToImageDomain) ? CLFFT_BACKWARD : CLFFT_FORWARD;

                // Command queue
                cl::CommandQueue &queue = device->get_execute_queue();

                // Events
                vector<cl::Event> inputReady(1);
                vector<cl::Event> fftFinished(1);
                vector<cl::Event> outputReady(1);

                // Device memory
                cl::Buffer d_grid = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof_grid());

                // Performance counter
                PerformanceCounter counter_fft;
                counter_fft.setPowerSensor(power_sensor);

                // Load kernel function
                unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                kernel_fft->shift(grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                double time_input = -omp_get_wtime();
                queue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid(), grid, NULL, &inputReady[0]);
                time_input += omp_get_wtime();

                // Create FFT plan
                #if ENABLE_FFT
                kernel_fft->plan(*context, queue, gridsize, 1);

				// Launch FFT
                kernel_fft->launchAsync(queue, d_grid, sign);
                #endif
                queue.enqueueMarkerWithWaitList(NULL, &fftFinished[0]);
                fftFinished[0].wait();

                // Copy grid to host
                double time_output = -omp_get_wtime();
                queue.enqueueReadBuffer(d_grid, CL_FALSE, 0, sizeof_grid(), grid, &fftFinished, &outputReady[0]);
                outputReady[0].wait();
                time_output += omp_get_wtime();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                kernel_fft->shift(grid);
                time_shift += omp_get_wtime();

                // Perform fft scaling
                double time_scale = -omp_get_wtime();
                complex<float> scale = complex<float>(2, 0);
                if (direction == FourierDomainToImageDomain) {
                    kernel_fft->scale(grid, scale);
                }
                time_scale += omp_get_wtime();

                #if defined(REPORT_TOTAL)
                auxiliary::report("   input", time_input, 0, sizeof_grid(), 0);
                auxiliary::report("     fft",
                                  PerformanceCounter::get_runtime((cl_event) inputReady[0](), (cl_event) fftFinished[0]()),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  0);
                auxiliary::report("  output", time_output, 0, sizeof_grid(), 0);
                auxiliary::report("fftshift", time_shift/2, 0, sizeof_grid() * 2, 0);
                if (direction == FourierDomainToImageDomain) {
                    auxiliary::report(" scaling", time_scale, 0, sizeof_grid() * 2, 0);
                }
                clog << endl;
                #endif
            } // end transform

        } // namespace opencl
    } // namespace proxy
} // namespace idg


// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::opencl::Generic OpenCL_Generic;

    OpenCL_Generic* OpenCL_Generic_init(
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

        return new OpenCL_Generic(P);
    }

    void OpenCL_Generic_grid(OpenCL_Generic* p,
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

    void OpenCL_Generic_degrid(OpenCL_Generic* p,
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

    void OpenCL_Generic_transform(OpenCL_Generic* p,
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

    void OpenCL_Generic_destroy(OpenCL_Generic* p) {
       delete p;
    }

} // end extern "C"
