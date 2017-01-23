#include <clFFT.h>

#include "Generic.h"

#include "DeviceInstance.h"
#include "PerformanceCounter.h"

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


using namespace std;
using namespace idg::kernel::opencl;


namespace idg {
    namespace proxy {
        namespace opencl {

            #if REDUCE_HOST_MEMORY
            std::vector<cl::Buffer*> h_visibilities_;
            std::vector<cl::Buffer*> h_uvw_;
            #else
            cl::Buffer *h_visibilities;
            cl::Buffer *h_uvw;
            #endif
            std::vector<cl::Buffer*> h_grid_;

            // Constructor
            Generic::Generic(
                CompileConstants constants) :
                OpenCL(constants)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                // Allocate memory
                //#if !REDUCE_HOST_MEMORY
                //h_visibilities = new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(params.get_nr_baselines()));
                //h_uvw = new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(params.get_nr_baselines()));
                //#endif
                //for (DeviceInstance *device : devices) {
                //    #if REDUCE_HOST_MEMORY
                //    h_visibilities_.push_back(new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(params.get_nr_baselines())));
                //    h_uvw_.push_back(new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(params.get_nr_baselines())));
                //    #endif
                //    h_grid_.push_back(new cl::Buffer(*context, CL_MEM_ALLOC_HOST_PTR, sizeof_grid()));
                //}

                // Initialize host PowerSensor
                #if defined(HAVE_LIKWID)
                hostPowerSensor = new LikwidPowerSensor();
                #else
                hostPowerSensor = new RaplPowerSensor();
                #endif
            }

            // Destructor
            Generic::~Generic() {
                //for (int i = 0; i < devices.size(); i++) {
                //    #if REDUCE_HOST_MEMORY
                //    delete h_visibilities_[i];
                //    delete h_uvw_[i];
                //    #endif
                //    delete h_grid_[i];
                //}
            }


            /* High level routines */
            void Generic::transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto grid_size = grid.get_x_dim();
                auto nr_correlations = mConstants.get_nr_correlations();

                // Load device
                DeviceInstance& device = get_device(0);
                cl::Context& context = get_context();
                PowerSensor *devicePowerSensor = device.get_powersensor();

                // Command queue
                cl::CommandQueue &queue = device.get_execute_queue();

                // Events
                vector<cl::Event> input(2);
                vector<cl::Event> output(2);

                // Performance counter
                PerformanceCounter counter;
                counter.setPowerSensor(devicePowerSensor);

                // Power measurement
                PowerSensor::State powerStates[4];
                powerStates[0] = hostPowerSensor->read();
                powerStates[2] = devicePowerSensor->read();

                // Device memory
                auto sizeof_grid = device.sizeof_grid(grid_size);
                cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid);

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                queue.enqueueMarkerWithWaitList(NULL, &input[0]);
                queue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid, grid.data());
                queue.enqueueMarkerWithWaitList(NULL, &input[1]);

                // Create FFT plan
                #if ENABLE_FFT
                device.plan_fft(grid_size, 1);
                #endif

				// Launch FFT
                #if ENABLE_FFT
                device.launch_fft(d_grid, direction, counter, "  grid-fft");
                #endif

                // Copy grid to host
                queue.enqueueMarkerWithWaitList(NULL, &output[0]);
                queue.enqueueReadBuffer(d_grid, CL_FALSE, 0, sizeof_grid, grid.data());
                queue.enqueueMarkerWithWaitList(NULL, &output[1]);
                output[1].wait();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Perform fft scaling
                double time_scale = -omp_get_wtime();
                complex<float> scale = complex<float>(2, 0);
                if (direction == FourierDomainToImageDomain) {
                    device.scale(grid, scale);
                }
                time_scale += omp_get_wtime();

                powerStates[1] = hostPowerSensor->read();
                powerStates[3] = devicePowerSensor->read();

                #if defined(REPORT_TOTAL)
                auxiliary::report("    input",
                                  PerformanceCounter::get_runtime((cl_event) input[0](), (cl_event) input[1]()),
                                  0, sizeof_grid, 0);
                auxiliary::report("   output",
                                  PerformanceCounter::get_runtime((cl_event) output[0](), (cl_event) output[1]()),
                                  0, sizeof_grid, 0);
                auxiliary::report("  fftshift", time_shift/2, 0, sizeof_grid * 2, 0);
                if (direction == FourierDomainToImageDomain) {
                auxiliary::report("grid-scale", time_scale, 0, sizeof_grid * 2, 0);
                }
                auxiliary::report("|host", 0, 0, hostPowerSensor, powerStates[0], powerStates[1]);
                auxiliary::report("|device", 0, 0, devicePowerSensor, powerStates[2], powerStates[3]);

                clog << endl;
                #endif
            } // end transform


            void Generic::gridding(
                const Plan& plan,
                const float w_offset, // in lambda
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Array3D<std::complex<float>>& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
#if 0
                // Configuration
                const int nr_devices = devices.size();
                const int nr_streams = 2;

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

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_streams);

                // Copy input data to host memory
                #if !REDUCE_HOST_MEMORY
                cl::CommandQueue &queue = devices[0]->get_htod_queue();
                writeBufferBatched(queue, *h_visibilities, CL_FALSE, 0, sizeof_visibilities(nr_baselines), visibilities);
                writeBufferBatched(queue, *h_uvw, CL_FALSE, 0, sizeof_uvw(nr_baselines), uvw);
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
                double total_runtime_gridding = 0;
                double time_gridding_start = 0;
                PowerSensor::State startStates[nr_devices+1];
                PowerSensor::State stopStates[nr_devices+1];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id = global_id % nr_streams;
                    int jobsize = jobsize_[device_id];

                    // Limit jobsize
                    jobsize = min(jobsize, nr_baselines); // Jobs can't be larger than number of baselines

                    // Load device
                    DeviceInstance *device    = devices[device_id];
                    PowerSensor *devicePowerSensor = device->get_powersensor();

                    // Load kernels
                    unique_ptr<Gridder> kernel_gridder = device->get_kernel_gridder();
                    unique_ptr<Adder>   kernel_adder   = device->get_kernel_adder();
                    unique_ptr<Scaler>  kernel_scaler  = device->get_kernel_scaler();
                    unique_ptr<GridFFT> kernel_fft     = device->get_kernel_fft();


                    // Load OpenCL objects
                    cl::CommandQueue &executequeue = device->get_execute_queue();
                    cl::CommandQueue &htodqueue    = device->get_htod_queue();
                    cl::CommandQueue &dtohqueue    = device->get_dtoh_queue();

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
                    }
                    array<float, 1> pattern;
                    pattern[0] = 0.;
                    htodqueue.enqueueFillBuffer(d_grid, pattern, 0, sizeof_grid());
                    htodqueue.finish();

                    // Events
                    vector<cl::Event> inputFree(1), outputFree(1), inputReady(1);
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
                    htodqueue.enqueueCopyBuffer(*h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize));
                    htodqueue.enqueueCopyBuffer(*h_visibilities, d_visibilities, 0, 0, sizeof_visibilities(jobsize));
                    htodqueue.finish();
                    #if ENABLE_FFT
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    #endif
                    executequeue.finish();
                    #endif

                    // Performance measurement
                    vector<PerformanceCounter> counters(5);
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(devicePowerSensor);
                    }
                    if (local_id == 0) {
                        startStates[device_id] = devicePowerSensor->read();
                    }
                    startStates[nr_devices] = hostPowerSensor->read();

                    #pragma omp barrier
                    #pragma omp single
                    time_gridding_start = omp_get_wtime();

                    for (int i = 0; i < nr_repetitions; i++) {
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
                        auto subgrid_offset       = plan.get_subgrid_offset(bl);

                        // Offsets
                        size_t uvw_offset          = bl * sizeof_uvw(1);
                        size_t visibilities_offset = bl * sizeof_visibilities(1);
                        size_t metadata_offset     = subgrid_offset * sizeof_metadata(1);

                        // Pointers to data for current batch
                        #if REDUCE_HOST_MEMORY
                        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw(current_nr_baselines), uvw_ptr);
                        htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0, sizeof_visibilities(current_nr_baselines), visibilities_ptr);
                        uvw_offset = 0;
                        visibilities_offset = 0;
                        #endif
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&inputFree, NULL);
                            htodqueue.enqueueCopyBuffer(*h_uvw, d_uvw, uvw_offset, 0, sizeof_uvw(current_nr_baselines));
                            htodqueue.enqueueCopyBuffer(*h_visibilities, d_visibilities, visibilities_offset, 0, sizeof_visibilities(current_nr_baselines));
                            htodqueue.enqueueWriteBuffer(d_metadata, CL_FALSE, 0, sizeof_metadata(current_nr_subgrids), metadata_ptr);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

							// Launch gridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            executequeue.enqueueMarkerWithWaitList(&outputFree, NULL);
                            kernel_gridder->launchAsync(
                                executequeue, current_nr_timesteps, current_nr_subgrids,
                                gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);

							// Launch FFT
                            #if ENABLE_FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD, counters[4], "sub-fft");
                            #endif

                            // Launch adder kernel
                            kernel_adder->launchAsync(
                                executequeue, current_nr_subgrids, gridsize,
                                d_metadata, d_subgrids, d_grid, counters[3]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);
                        }

                        inputReady[0].wait();
                    } // end for bl
                    } // end for repetitions

                    // Wait for all jobs to finish
                    executequeue.finish();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = devicePowerSensor->read();
                        stopStates[nr_devices] = hostPowerSensor->read();
                    }
                } // end omp parallel

                // End gridding timing
                #pragma omp critical
                {
                    total_runtime_gridding = (omp_get_wtime() - time_gridding_start) / nr_repetitions;
                }

                // Add new grids to existing grid
                for (int d = 0; d < devices.size(); d++) {
                    cl::Buffer &d_grid = *(d_grid_[d]);
                    float2 grid_src[gridsize * gridsize * nr_polarizations];
                    queue.enqueueReadBuffer(d_grid, CL_TRUE, 0, sizeof_grid(), grid_src);
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
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);

                // Report host power consumption
                auxiliary::report("|host", 0, 0, hostPowerSensor, startStates[nr_devices], stopStates[nr_devices]);

                // Report device power consumption
                for (int d = 0; d < devices.size(); d++) {
                    PowerSensor *devicePowerSensor = devices[d]->get_powersensor();
                    stringstream message;
                    message << "|device" << d;
                    auxiliary::report(message.str().c_str(), 0, 0, devicePowerSensor, startStates[d], stopStates[d]);
                }
                clog << endl;
                #endif
#endif
            } // end gridding


            void Generic::degridding(
                const Plan& plan,
                const float w_offset, // in lambda
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array3D<std::complex<float>>& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
#if 0
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

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_streams);

                // Host memory
                cl::Buffer &h_grid = *(h_grid_[0]);
                cl::CommandQueue &queue = devices[0]->get_htod_queue();
                queue.enqueueWriteBuffer(h_grid, CL_FALSE, 0, sizeof_grid(), grid);
                #if !REDUCE_HOST_MEMORY
                queue.enqueueWriteBuffer(*h_uvw, CL_FALSE, 0, sizeof_uvw(nr_baselines), uvw);
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
                PowerSensor::State startStates[nr_devices+1];
                PowerSensor::State stopStates[nr_devices+1];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id = global_id % nr_streams;
                    int jobsize = jobsize_[device_id];

                    // Limit jobsize
                    jobsize = min(jobsize, nr_baselines); // Jobs can't be larger than number of baselines
                    jobsize = min(jobsize, 256); // TODO: larger values cause performance degradation

                    // Load device
                    DeviceInstance *device    = devices[device_id];
                    PowerSensor *devicePowerSensor = device->get_powersensor();

                    // Load kernels
                    unique_ptr<Degridder> kernel_degridder = device->get_kernel_degridder();
                    unique_ptr<Splitter>  kernel_splitter  = device->get_kernel_splitter();;
                    unique_ptr<GridFFT>   kernel_fft       = device->get_kernel_fft();

                    // Load OpenCL objects
                    cl::CommandQueue &executequeue = device->get_execute_queue();
                    cl::CommandQueue &htodqueue    = device->get_htod_queue();
                    cl::CommandQueue &dtohqueue    = device->get_dtoh_queue();

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
                        htodqueue.enqueueCopyBuffer(h_grid, d_grid, 0, 0, sizeof_grid());
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
                    htodqueue.enqueueCopyBuffer(*h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize));
                    void *metadata_ptr     = (void *) plan.get_metadata_ptr(0);
                    htodqueue.enqueueWriteBuffer(d_metadata, CL_FALSE, 0, sizeof_metadata(max_nr_subgrids), metadata_ptr);
                    htodqueue.finish();
                    #if ENABLE_FFT
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    executequeue.finish();
                    #endif
                    #endif

                    // Performance measurement
                    vector<PerformanceCounter> counters(4);
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(devicePowerSensor);
                    }
                    if (local_id == 0) {
                        startStates[device_id] = devicePowerSensor->read();
                    }
                    startStates[nr_devices] = hostPowerSensor->read();

                    #pragma omp barrier
                    #pragma omp single
                    time_degridding_start = omp_get_wtime();

                    for (int i = 0; i < nr_repetitions; i++) {
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
                            htodqueue.enqueueCopyBuffer(*h_uvw, d_uvw, uvw_offset, 0, sizeof_uvw(current_nr_baselines));
                            htodqueue.enqueueWriteBuffer(d_metadata, CL_FALSE, 0, sizeof_metadata(current_nr_subgrids), metadata_ptr);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch splitter kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            kernel_splitter->launchAsync(
                                executequeue, current_nr_subgrids, gridsize,
                                d_metadata, d_subgrids, d_grid, counters[0]);

                            // Launch FFT
                            #if ENABLE_FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_FORWARD, counters[3], "sub-fft");
                            #endif

                            // Launch degridder kernel
                            executequeue.enqueueMarkerWithWaitList(&outputFree, NULL);
                            kernel_degridder->launchAsync(
                                executequeue, current_nr_timesteps, current_nr_subgrids,
                                gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[2]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &inputFree[0]);

                            // Copy visibilities to host
                            dtohqueue.enqueueMarkerWithWaitList(&outputReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, *h_visibilities, 0, visibilities_offset, sizeof_visibilities(current_nr_baselines), NULL, &outputFree[0]);
                        }

                        #if REDUCE_HOST_MEMORY
                        outputFree[0].wait();
                        dtohqueue.enqueueReadBuffer(h_visibilities, CL_FALSE, 0, sizeof_visibilities(current_nr_baselines), visibilities_ptr);
                        #endif

                        inputFree[0].wait();
                    } // end for bl
                    } // end for repetitions

                    // Wait for all jobs to finish
                    dtohqueue.finish();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = devicePowerSensor->read();
                        stopStates[nr_devices] = hostPowerSensor->read();
                    }
                } // end omp parallel

                // End degridding timing
                #pragma omp critical
                {
                    total_runtime_degridding = (omp_get_wtime() - time_degridding_start) / nr_repetitions;
                }

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
                    queue.enqueueReadBuffer(*h_visibilities, CL_FALSE, offset, sizeof_visibilities(1), visibilities_ptr);
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

                // Report host power consumption
                auxiliary::report("|host", 0, 0, hostPowerSensor, startStates[nr_devices], stopStates[nr_devices]);

                // Report device power consumption
                for (int d = 0; d < devices.size(); d++) {
                    PowerSensor *devicePowerSensor = devices[d]->get_powersensor();
                    stringstream message;
                    message << "|device" << d;
                    auxiliary::report(message.str().c_str(), 0, 0, devicePowerSensor, startStates[d], stopStates[d]);
                }
                clog << endl;
                #endif
#endif
            } // end degridding

        } // namespace opencl
    } // namespace proxy
} // namespace idg
