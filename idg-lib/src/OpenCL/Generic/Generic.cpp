#include <clFFT.h>

#include "Generic.h"

#include "InstanceOpenCL.h"
#include "PerformanceCounter.h"

/*
    Toggle warmup
        Copy some memory to device prior to
        starting the actual computation
*/
#define ENABLE_WARMUP 0

/*
    Toggle planning and execution of Fourier transformations on and off
        The clFFT library contains memory leaks, which makes it much harder
        to find and resolve issues in non-library code. This option disables
        usage of the library so that they can be resolved
*/
#define ENABLE_FFT 1


using namespace std;
using namespace idg::kernel::opencl;


namespace idg {
    namespace proxy {
        namespace opencl {

            // Constructor
            Generic::Generic(
                CompileConstants& constants) :
                OpenCL(constants)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                // Initialize host PowerSensor
                #if defined(HAVE_LIKWID)
                hostPowerSensor = LikwidPowerSensor::create();
                #else
                hostPowerSensor = RaplPowerSensor::create();
                #endif
            }

            // Destructor
            Generic::~Generic() {
                delete hostPowerSensor;
            }


            /* High level routines */
            void Generic::do_transform(
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
                InstanceOpenCL& device = get_device(0);
                cl::Context& context   = get_context();
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


            void Generic::do_gridding(
                const Plan& plan,
                const float w_step, // in lambda
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Proxy constants
                auto subgrid_size    = mConstants.get_subgrid_size();
                auto nr_correlations = mConstants.get_nr_correlations();

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                check_dimensions(
                    frequencies, visibilities, uvw, baselines,
                    grid, aterms, aterms_offsets, spheroidal);

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto nr_stations  = aterms.get_z_dim();
                auto nr_timeslots = aterms.get_w_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 2;

                // Initialize metadata
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, nr_streams);

                // Initialize memory for first device
                cl::Context& context        = get_context();
                InstanceOpenCL& device0     = get_device(0);
                cl::CommandQueue& htodqueue = device0.get_htod_queue();
                auto sizeof_visibilities    = device0.sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                auto sizeof_uvw             = device0.sizeof_uvw(nr_baselines, nr_timesteps);
                cl::Buffer h_visibilities   = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities);
                cl::Buffer h_uvw            = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw);
                htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0, sizeof_visibilities, visibilities.data());
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw, uvw.data());

                // Initialize memory for all devices
                std::vector<cl::Buffer> h_grid_(nr_devices);
                std::vector<cl::Buffer> d_grid_(nr_devices);
                std::vector<cl::Buffer> d_wavenumbers_(nr_devices);
                std::vector<cl::Buffer> d_spheroidal_(nr_devices);
                std::vector<cl::Buffer> d_aterms_(nr_devices);
                for (int d = 0; d < nr_devices; d++) {
                    vector<cl::Event> input(2);
                    InstanceOpenCL& device      = get_device(d);
                    cl::CommandQueue& htodqueue = device.get_htod_queue();
                    auto sizeof_grid         = device.sizeof_grid(grid_size);
                    auto sizeof_wavenumbers  = device.sizeof_wavenumbers(nr_channels);
                    auto sizeof_spheroidal   = device.sizeof_spheroidal(subgrid_size);
                    auto sizeof_aterms       = device.sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                    cl::Buffer h_grid        = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof_grid);
                    cl::Buffer d_grid        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid);
                    cl::Buffer d_wavenumbers = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_wavenumbers);
                    cl::Buffer d_spheroidal  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_spheroidal);
                    cl::Buffer d_aterms      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_aterms);
                    htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, sizeof_wavenumbers, wavenumbers.data());
                    htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, sizeof_spheroidal, spheroidal.data());
                    htodqueue.enqueueWriteBuffer(d_aterms, CL_FALSE, 0, sizeof_aterms, aterms.data());
                    htodqueue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid, grid.data());
                    h_grid_[d]        = h_grid;
                    d_grid_[d]        = d_grid;
                    d_wavenumbers_[d] = d_wavenumbers;
                    d_spheroidal_[d]  = d_spheroidal;
                    d_aterms_[d]      = d_aterms;
                }

                // Locks
                int locks[nr_devices];

                // Performance measurements
                double total_runtime_gridding = 0;
                PowerSensor::State startStates[nr_devices+1];
                PowerSensor::State stopStates[nr_devices+1];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = jobsize_[device_id];
                    int lock      = locks[device_id];
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                    // Limit jobsize
                    jobsize = min(jobsize, nr_baselines);

                    // Load device
                    InstanceOpenCL& device = get_device(device_id);

                    // Load OpenCL objects
                    cl::CommandQueue& executequeue = device.get_execute_queue();
                    cl::CommandQueue& htodqueue    = device.get_htod_queue();
                    cl::CommandQueue& dtohqueue    = device.get_dtoh_queue();

                    // Load memory objects
                    cl::Buffer& h_grid        = h_grid_[device_id];
                    cl::Buffer& d_grid        = d_grid_[device_id];
                    cl::Buffer& d_wavenumbers = d_wavenumbers_[device_id];
                    cl::Buffer& d_spheroidal  = d_spheroidal_[device_id];
                    cl::Buffer& d_aterms      = d_aterms_[device_id];

                    // Events
                    vector<cl::Event> inputFree(1), outputFree(1), inputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &inputFree[0]);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);

                    // Allocate private device memory
                    auto sizeof_visibilities = device.sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
                    auto sizeof_uvw          = device.sizeof_uvw(jobsize, nr_timesteps);
                    auto sizeof_subgrids     = device.sizeof_subgrids(max_nr_subgrids, subgrid_size);
                    auto sizeof_metadata     = device.sizeof_metadata(max_nr_subgrids);
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_visibilities);
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_uvw);
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids);
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata);

                    // Create FFT plan
                    #if ENABLE_FFT
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
                    }
                    #endif

                    // Warmup
                    #if ENABLE_WARMUP
                    htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, 0, 0, sizeof_visibilities);
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw);
                    htodqueue.finish();
                    #endif

                    // Performance measurement
                    PowerSensor *devicePowerSensor = device.get_powersensor();
                    std::vector<PerformanceCounter> counters(3);
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(devicePowerSensor);
                    }
                    if (local_id == 0) {
                        startStates[device_id] = devicePowerSensor->read();
                    }
                    startStates[nr_devices] = hostPowerSensor->read();

                    #pragma omp barrier
                    #pragma omp single
                    total_runtime_gridding = -omp_get_wtime();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        unsigned int first_bl, last_bl, current_nr_baselines;
                        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                        if (current_nr_baselines == 0) continue;

                        // Initialize iteration
                        auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        auto subgrid_offset       = plan.get_subgrid_offset(first_bl);
                        auto uvw_offset           = first_bl * device.sizeof_uvw(1, nr_timesteps);
                        auto visibilities_offset  = first_bl * device.sizeof_visibilities(1, nr_timesteps, nr_channels);
                        auto *metadata_ptr        = (void *) plan.get_metadata_ptr(first_bl);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&inputFree, NULL);
                            htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0,
                                device.sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0,
                                device.sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodqueue.enqueueWriteBuffer(d_metadata, CL_FALSE, 0,
                                device.sizeof_metadata(current_nr_subgrids), metadata_ptr);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

							// Launch gridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            executequeue.enqueueMarkerWithWaitList(&outputFree, NULL);
                            device.launch_gridder(
                                current_nr_timesteps, current_nr_subgrids,
                                grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                                d_aterms, d_metadata, d_subgrids, counters[0]);

							// Launch FFT
                            #if ENABLE_FFT
                            device.launch_fft(d_subgrids, FourierDomainToImageDomain, counters[1], "sub-fft");
                            #endif

                            // Launch adder kernel
                            device.launch_adder(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, d_grid, counters[2]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);
                        }

                        inputReady[0].wait();
                    } // end for bl

                    // Wait for all jobs to finish
                    executequeue.finish();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = devicePowerSensor->read();
                        stopStates[nr_devices] = hostPowerSensor->read();
                    }

                    // Copy grid to host
                    if (local_id == 0) {
                        dtohqueue.enqueueCopyBuffer(d_grid, h_grid, 0, 0, device.sizeof_grid(grid_size));
                    }
                    dtohqueue.finish();
                } // end omp parallel

                // End timing
                total_runtime_gridding += omp_get_wtime();

                // Add grids
                float2 grid_src[nr_correlations * grid_size * grid_size];
                for (int d = 0; d < nr_devices; d++) {
                    InstanceOpenCL& device     = get_device(d);
                    cl::CommandQueue dtohqueue = device.get_dtoh_queue();
                    dtohqueue.enqueueReadBuffer(h_grid_[d], CL_TRUE, 0, device.sizeof_grid(grid_size), grid_src);
                    float2 *grid_dst = (float2 *) grid.data();

                    #pragma omp parallel for
                    for (int i = 0; i < grid_size * grid_size * nr_correlations; i++) {
                        grid_dst[i] += grid_src[i];
                    }
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                InstanceOpenCL& device        = get_device(0);
                auto total_nr_subgrids        = plan.get_nr_subgrids();
                auto total_nr_timesteps       = plan.get_nr_timesteps();
                auto total_nr_visibilities    = plan.get_nr_visibilities();
                uint64_t total_flops_gridder  = device.flops_gridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_gridder  = device.bytes_gridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft      = device.flops_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_bytes_fft      = device.bytes_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_flops_scaler   = device.flops_scaler(total_nr_subgrids);
                uint64_t total_bytes_scaler   = device.bytes_scaler(total_nr_subgrids);
                uint64_t total_flops_adder    = device.flops_adder(total_nr_subgrids);
                uint64_t total_bytes_adder    = device.bytes_adder(total_nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft + total_flops_scaler + total_flops_adder;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft + total_bytes_scaler + total_bytes_adder;
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, total_nr_visibilities);

                // Report host power consumption
                auxiliary::report("|host", 0, 0, hostPowerSensor, startStates[nr_devices], stopStates[nr_devices]);

                // Report device power consumption
                for (int d = 0; d < get_num_devices(); d++) {
                    PowerSensor* devicePowerSensor = get_device(d).get_powersensor();
                    stringstream message;
                    message << "|device" << d;
                    auxiliary::report(message.str().c_str(), 0, 0, devicePowerSensor, startStates[d], stopStates[d]);
                }
                clog << endl;
                #endif
            } // end gridding


            void Generic::do_degridding(
                const Plan& plan,
                const float w_step, // in lambda
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Proxy constants
                auto subgrid_size    = mConstants.get_subgrid_size();
                auto nr_correlations = mConstants.get_nr_correlations();

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                check_dimensions(
                    frequencies, visibilities, uvw, baselines,
                    grid, aterms, aterms_offsets, spheroidal);

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto nr_stations  = aterms.get_z_dim();
                auto nr_timeslots = aterms.get_w_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 2;

                // Initialize metadata
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, nr_streams);

                // Initialize memory for first device
                cl::Context& context        = get_context();
                InstanceOpenCL& device0     = get_device(0);
                cl::CommandQueue& htodqueue = device0.get_htod_queue();
                auto sizeof_visibilities    = device0.sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                auto sizeof_uvw             = device0.sizeof_uvw(nr_baselines, nr_timesteps);
                cl::Buffer h_visibilities   = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities);
                cl::Buffer h_uvw            = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw);
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw, uvw.data());

                // Initialize memory for all devices
                std::vector<cl::Buffer> d_grid_(nr_devices);
                std::vector<cl::Buffer> d_wavenumbers_(nr_devices);
                std::vector<cl::Buffer> d_spheroidal_(nr_devices);
                std::vector<cl::Buffer> d_aterms_(nr_devices);
                for (int d = 0; d < nr_devices; d++) {
                    vector<cl::Event> input(2);
                    InstanceOpenCL& device      = get_device(d);
                    cl::CommandQueue& htodqueue = device.get_htod_queue();
                    auto sizeof_grid         = device.sizeof_grid(grid_size);
                    auto sizeof_wavenumbers  = device.sizeof_wavenumbers(nr_channels);
                    auto sizeof_spheroidal   = device.sizeof_spheroidal(subgrid_size);
                    auto sizeof_aterms       = device.sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                    cl::Buffer d_grid        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid);
                    cl::Buffer d_wavenumbers = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_wavenumbers);
                    cl::Buffer d_spheroidal  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_spheroidal);
                    cl::Buffer d_aterms      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_aterms);
                    htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, sizeof_wavenumbers, wavenumbers.data());
                    htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, sizeof_spheroidal, spheroidal.data());
                    htodqueue.enqueueWriteBuffer(d_aterms, CL_FALSE, 0, sizeof_aterms, aterms.data());
                    htodqueue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid, grid.data());
                    d_grid_[d]        = d_grid;
                    d_wavenumbers_[d] = d_wavenumbers;
                    d_spheroidal_[d]  = d_spheroidal;
                    d_aterms_[d]      = d_aterms;
                }

                // Locks
                int locks[nr_devices];

                // Performance measurements
                double total_runtime_degridding = 0;
                PowerSensor::State startStates[nr_devices+1];
                PowerSensor::State stopStates[nr_devices+1];

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = jobsize_[device_id];
                    int lock      = locks[device_id];
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                    // Limit jobsize
                    jobsize = min(jobsize, nr_baselines);

                    // Load device
                    InstanceOpenCL& device = get_device(device_id);

                    // Load OpenCL objects
                    cl::CommandQueue& executequeue = device.get_execute_queue();
                    cl::CommandQueue& htodqueue    = device.get_htod_queue();
                    cl::CommandQueue& dtohqueue    = device.get_dtoh_queue();

                    // Load memory objects
                    cl::Buffer& d_grid        = d_grid_[device_id];
                    cl::Buffer& d_wavenumbers = d_wavenumbers_[device_id];
                    cl::Buffer& d_spheroidal  = d_spheroidal_[device_id];
                    cl::Buffer& d_aterms      = d_aterms_[device_id];

                    // Events
                    vector<cl::Event> inputFree(1), outputFree(1), inputReady(1), outputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &inputFree[0]);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);

                    // Allocate private device memory
                    auto sizeof_visibilities = device.sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
                    auto sizeof_uvw          = device.sizeof_uvw(jobsize, nr_timesteps);
                    auto sizeof_subgrids     = device.sizeof_subgrids(max_nr_subgrids, subgrid_size);
                    auto sizeof_metadata     = device.sizeof_metadata(max_nr_subgrids);
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_visibilities);
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_uvw);
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids);
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata);

                    // Create FFT plan
                    #if ENABLE_FFT
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
                    }
                    #endif

                    // Warmup
                    #if ENABLE_WARMUP
                    htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, 0, 0, sizeof_visibilities);
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw);
                    htodqueue.finish();
                    #endif

                    // Performance measurement
                    PowerSensor *devicePowerSensor = device.get_powersensor();
                    vector<PerformanceCounter> counters(3);
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(devicePowerSensor);
                    }
                    if (local_id == 0) {
                        startStates[device_id] = devicePowerSensor->read();
                    }
                    startStates[nr_devices] = hostPowerSensor->read();

                    #pragma omp barrier
                    #pragma omp single
                    total_runtime_degridding = -omp_get_wtime();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        unsigned int first_bl, last_bl, current_nr_baselines;
                        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                        if (current_nr_baselines == 0) continue;

                        // Initialize iteration
                        auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        auto subgrid_offset       = plan.get_subgrid_offset(first_bl);
                        auto uvw_offset           = first_bl * device.sizeof_uvw(1, nr_timesteps);
                        auto visibilities_offset  = first_bl * device.sizeof_visibilities(1, nr_timesteps, nr_channels);
                        void *metadata_ptr        = (void *) plan.get_metadata_ptr(first_bl);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&inputFree, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0,
                                device.sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodqueue.enqueueWriteBuffer(d_metadata, CL_FALSE, 0,
                                device.sizeof_metadata(current_nr_subgrids), metadata_ptr);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch splitter kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            device.launch_splitter(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, d_grid, counters[0]);

                            // Launch FFT
                            #if ENABLE_FFT
                            device.launch_fft(d_subgrids, ImageDomainToFourierDomain, counters[1], "sub-fft");
                            #endif

                            // Launch degridder kernel
                            executequeue.enqueueMarkerWithWaitList(&outputFree, NULL);
                            device.launch_degridder(
                                current_nr_timesteps, current_nr_subgrids,
                                grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                                d_aterms, d_metadata, d_subgrids, counters[2]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &inputFree[0]);

                            // Copy visibilities to host
                            dtohqueue.enqueueMarkerWithWaitList(&outputReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset,
                                device.sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels),
                                NULL, &outputFree[0]);
                        }

                        inputFree[0].wait();
                    } // end for bl

                    // Wait for all jobs to finish
                    dtohqueue.finish();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = devicePowerSensor->read();
                        stopStates[nr_devices] = hostPowerSensor->read();
                    }

                } // end omp parallel

                // End timing
                total_runtime_degridding += omp_get_wtime();

                // Copy visibilities
                cl::CommandQueue& dtohqueue = device0.get_dtoh_queue();
                dtohqueue.enqueueReadBuffer(h_visibilities, CL_TRUE, 0,
                    device0.sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels), visibilities.data());

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                InstanceOpenCL& device          = get_device(0);
                auto total_nr_subgrids          = plan.get_nr_subgrids();
                auto total_nr_timesteps         = plan.get_nr_timesteps();
                auto total_nr_visibilities      = plan.get_nr_visibilities();
                uint64_t total_flops_splitter   = device.flops_splitter(total_nr_subgrids);
                uint64_t total_bytes_splitter   = device.bytes_splitter(total_nr_subgrids);
                uint64_t total_flops_fft        = device.flops_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_bytes_fft        = device.bytes_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_flops_degridder  = device.flops_degridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_degridder  = device.bytes_degridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, total_nr_visibilities);

                // Report host power consumption
                auxiliary::report("|host", 0, 0, hostPowerSensor, startStates[nr_devices], stopStates[nr_devices]);

                // Report device power consumption
                for (int d = 0; d < get_num_devices(); d++) {
                    PowerSensor* devicePowerSensor = get_device(d).get_powersensor();
                    stringstream message;
                    message << "|device" << d;
                    auxiliary::report(message.str().c_str(), 0, 0, devicePowerSensor, startStates[d], stopStates[d]);
                }
                clog << endl;
                #endif
            } // end degridding

        } // namespace opencl
    } // namespace proxy
} // namespace idg

#include "GenericC.h"
