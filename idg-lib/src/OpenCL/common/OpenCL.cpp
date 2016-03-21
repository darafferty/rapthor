#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <fstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()
#include <unistd.h> // rmdir()

#include "idg-config.h"
#include "OpenCL.h"

#define WARMUP 1

using namespace std;
using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            /// Constructors
            OpenCL::OpenCL(
                Parameters params,
                unsigned deviceNumber,
                Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                // Create context
                context = cl::Context(CL_DEVICE_TYPE_ALL);

            	// Get devices
            	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
                device = devices[deviceNumber];

                // Set/check parameters
                mParams = params;
                parameter_sanity_check(); // throws exception if bad parameters

                // Compile kernels
                compile(flags);

                // Initialize clFFT
                clfftSetupData setup;
                clfftSetup(&setup);

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

            OpenCL::~OpenCL()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (int i = 0; i < programs.size(); i++) {
                    delete programs[i];
                }

                clfftTeardown();
            }

            string OpenCL::default_compiler_flags() {
                return "-cl-fast-relaxed-math -cl-std=CL2.0";
            }


            /* Sizeof routines */
            uint64_t OpenCL::sizeof_subgrids(int nr_subgrids) {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_uvw(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                return 1ULL * nr_baselines * nr_time * sizeof(UVW);
            }

            uint64_t OpenCL::sizeof_visibilities(int nr_baselines) {
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                return 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_metadata(int nr_subgrids) {
                return 1ULL * nr_subgrids * sizeof(Metadata);
            }

            uint64_t OpenCL::sizeof_grid() {
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                return 1ULL * nr_polarizations * gridsize * gridsize * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_wavenumbers() {
                auto nr_channels = mParams.get_nr_channels();
                return 1ULL * nr_channels * sizeof(float);
            }

            uint64_t OpenCL::sizeof_aterm() {
                auto nr_stations = mParams.get_nr_stations();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>);
            }

            uint64_t OpenCL::sizeof_spheroidal() {
                auto subgridsize = mParams.get_subgrid_size();
                return 1ULL * subgridsize * subgridsize * sizeof(complex<float>);
            }


            /* High level routines */
            void OpenCL::grid_visibilities(
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
                jobsize = nr_baselines < jobsize ? nr_baselines : jobsize;

                // Load kernels
                unique_ptr<Gridder> kernel_gridder = get_kernel_gridder();
                unique_ptr<Adder> kernel_adder = get_kernel_adder();
                unique_ptr<Scaler> kernel_scaler = get_kernel_scaler();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue htodqueue    = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue dtohqueue    = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                const int nr_streams = 3;

                // Host memory
                cl::Buffer h_visibilities(context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(context, CL_MEM_ALLOC_HOST_PTR, sizeof_metadata(plan.get_nr_subgrids()));

                // Copy input data to host memory
                htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0,  sizeof_visibilities(nr_baselines), visibilities);
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0,  sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0,  sizeof_metadata(nr_subgrids), metadata);

                // Device memory
                cl::Buffer d_wavenumbers = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_wavenumbers());
                cl::Buffer d_aterm       = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_aterm());
                cl::Buffer d_spheroidal  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_spheroidal());
                cl::Buffer d_grid        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid());

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
                kernel_fft->plan(context, executequeue, subgridsize, max_nr_subgrids);

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Private device memory
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_visibilities(jobsize));
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata(max_nr_subgrids));

                    // Warmup
                    #if WARMUP
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, 0, 0, sizeof_metadata(max_nr_subgrids), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, 0, 0, sizeof_visibilities(jobsize), NULL, NULL);
                    htodqueue.finish();
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    executequeue.finish();
                    #endif

                    // Events
                    vector<cl::Event> inputReady(1), outputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);

                    // Performance counters
                    vector<PerformanceCounter> counters(4);
                    #if defined(MEASURE_POWER_ARDUINO)
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(&powerSensor);
                    }
                    #endif
                    #pragma omp single
                    startState = powerSensor.read();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Offsets
                        size_t uvw_offset          = bl * sizeof_uvw(1);
                        size_t visibilities_offset = bl * sizeof_visibilities(1);
                        size_t metadata_offset     = bl * sizeof_metadata(1);

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
                                executequeue, current_nr_baselines, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);

        					// Launch FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);

                            // Launch scaler kernel
                            kernel_scaler->launchAsync(executequeue, current_nr_subgrids, d_subgrids, counters[2]);

                            // Launch adder kernel
                            kernel_adder->launchAsync(executequeue, current_nr_subgrids, d_metadata, d_subgrids, d_grid, counters[3]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
                        }
                    }

                    outputReady[0].wait();
                }

                // Copy grid to host
                executequeue.finish();
                PowerSensor::State stopState = powerSensor.read();
                dtohqueue.enqueueReadBuffer(d_grid, CL_TRUE, 0, sizeof_grid(), grid, NULL, NULL);
                dtohqueue.finish();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
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
                double total_runtime_gridding = PowerSensor::seconds(startState, stopState);
                double total_watt_gridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            } // grid_visibilities


            void OpenCL::degrid_visibilities(
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

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_degridder();
                jobsize = nr_baselines < jobsize ? nr_baselines : jobsize;

                // Load kernels
                unique_ptr<Degridder> kernel_degridder = get_kernel_degridder();
                unique_ptr<Splitter> kernel_splitter = get_kernel_splitter();;
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue htodqueue    = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue dtohqueue    = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                const int nr_streams = 3;

                // Host memory
                cl::Buffer h_visibilities(context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(context, CL_MEM_ALLOC_HOST_PTR, sizeof_metadata(plan.get_nr_subgrids()));

                // Copy input data to host memory
                htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0,  sizeof_visibilities(nr_baselines), visibilities);
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0,  sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0,  sizeof_metadata(nr_subgrids), metadata);

                // Device memory
                cl::Buffer d_wavenumbers = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_wavenumbers());
                cl::Buffer d_aterm       = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_aterm());
                cl::Buffer d_spheroidal  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_spheroidal());
                cl::Buffer d_grid        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid());

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_adder = 0;
                PowerSensor::State startState;

                // Copy static device memory
                htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, sizeof_wavenumbers(), wavenumbers);
                htodqueue.enqueueWriteBuffer(d_aterm, CL_FALSE, 0, sizeof_aterm(), aterm);
                htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, sizeof_spheroidal(), spheroidal);
                htodqueue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid(), grid);
                htodqueue.finish();

                // Initialize fft
                auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                kernel_fft->plan(context, executequeue, subgridsize, max_nr_subgrids);

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Private device memory
                    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_visibilities(jobsize));
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE,  sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata(max_nr_subgrids));

                    // Warmup
                    #if WARMUP
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, 0, 0, sizeof_metadata(max_nr_subgrids), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, 0, 0, sizeof_visibilities(jobsize), NULL, NULL);
                    htodqueue.finish();
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    executequeue.finish();
                    #endif

                    // Events
                    vector<cl::Event> inputReady(1), computeReady(1), outputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);

                    // Performance counters
                    vector<PerformanceCounter> counters(3);
                    #if defined(MEASURE_POWER_ARDUINO)
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(&powerSensor);
                    }
                    #endif
                    #pragma omp single
                    startState = powerSensor.read();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Offsets
                        size_t uvw_offset          = bl * sizeof_uvw(1);
                        size_t visibilities_offset = bl * sizeof_visibilities(1);
                        size_t metadata_offset     = bl * sizeof_metadata(1);

                        #pragma omp critical (GPU)
                        {
        					// Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&outputReady, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, sizeof_uvw(current_nr_baselines), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, sizeof_metadata(current_nr_subgrids), NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch splitter kernel
                            executequeue.enqueueBarrierWithWaitList(&inputReady, NULL);
                            kernel_splitter->launchAsync(executequeue, current_nr_subgrids, d_metadata, d_subgrids, d_grid, counters[0]);

        					// Launch FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_FORWARD);

        					// Launch degridder kernel
                            kernel_degridder->launchAsync(
                                executequeue, current_nr_baselines, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[2]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

        					// Copy visibilities to host
                            dtohqueue.enqueueBarrierWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, sizeof_visibilities(current_nr_baselines), NULL, &outputReady[0]);
                        }
                    }

                    outputReady[0].wait();
                }

                // Copy visibilities
                dtohqueue.finish();
                PowerSensor::State stopState = powerSensor.read();
                dtohqueue.enqueueReadBuffer(h_visibilities, CL_TRUE, 0, sizeof_visibilities(nr_baselines), visibilities, NULL, NULL);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_splitter   = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                double total_runtime_degridding = PowerSensor::seconds(startState, stopState);
                double total_watt_degridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
           } // degrid_visibilities


            void OpenCL::transform(
                DomainAtoDomainB direction,
                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                clfftDirection sign = (direction == FourierDomainToImageDomain) ? CLFFT_BACKWARD : CLFFT_FORWARD;

                // Command queue
                cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

                // Events
                vector<cl::Event> events(4);

                // Host memory
                cl::Buffer h_grid(context, CL_MEM_READ_WRITE, sizeof_grid());
                queue.enqueueWriteBuffer(h_grid, CL_FALSE, 0, sizeof_grid(), grid);

                // Device memory
                cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid());

                // Performance counter
                PerformanceCounter counter_fft;
                #if defined(MEASURE_POWER_ARDUINO)
                counter_fft.setPowerSensor(&powerSensor);
                #endif

                // Load kernel function
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Copy grid to device
                queue.enqueueCopyBuffer(h_grid, d_grid, 0, 0, sizeof_grid(), NULL, &events[0]);

                // Create FFT plan
                kernel_fft->plan(context, queue, gridsize, 1);

        		// Launch FFT
                queue.enqueueMarkerWithWaitList(NULL, &events[1]);
                kernel_fft->launchAsync(queue, d_grid, sign);
                queue.enqueueMarkerWithWaitList(NULL, &events[2]);

                // Copy grid to host
                queue.enqueueCopyBuffer(d_grid, h_grid, 0, 0, sizeof_grid(), NULL, &events[3]);
                queue.enqueueReadBuffer(h_grid, CL_TRUE, 0, sizeof_grid(), grid);

                // Wait for fft to finish
                queue.finish();

                // Perform fft shift
                double runtime = -omp_get_wtime();
                kernel_fft->shift(grid);
                runtime += omp_get_wtime();

                #if defined(REPORT_TOTAL)
                auxiliary::report("     fft",
                                  PerformanceCounter::get_runtime((cl_event) events[1](), (cl_event) events[2]()),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  0);
                auxiliary::report("fftshift", runtime, 0, sizeof_grid() * 2, 0);
                clog << endl;
                #endif
            } // transform

            void OpenCL::compile(Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Source directory
                stringstream _srcdir;
                _srcdir << string(IDG_INSTALL_DIR);
                _srcdir << "/lib/kernels/OpenCL";
                string srcdir = _srcdir.str();

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
                string mparameters = Parameters::definitions(
                    mParams.get_nr_stations(),
                    mParams.get_nr_baselines(),
                    mParams.get_nr_channels(),
                    mParams.get_nr_time(),
                    mParams.get_nr_timeslots(),
                    mParams.get_imagesize(),
                    mParams.get_nr_polarizations(),
                    mParams.get_grid_size(),
                    mParams.get_subgrid_size());

                // Build parameters tring
                stringstream _parameters;
                _parameters << " " << flags;
                _parameters << " " << "-I " << srcdir;
                _parameters << " " << mparameters;
                string parameters = _parameters.str();

                // Create vector of devices
                std::vector<cl::Device> devices;
                devices.push_back(device);

                // Add all kernels to build
                vector<string> v;
                v.push_back("KernelGridder.cl");
                v.push_back("KernelDegridder.cl");
                v.push_back("KernelAdder.cl");
                v.push_back("KernelSplitter.cl");
                v.push_back("KernelScaler.cl");

                // Build OpenCL programs
                for (int i = 0; i < v.size(); i++) {
                    // Get source filename
                    stringstream _source_file_name;
                    _source_file_name << srcdir << "/" << v[i];
                    string source_file_name = _source_file_name.str();

                    // Read source from file
                    ifstream source_file(source_file_name.c_str());
                    string source(std::istreambuf_iterator<char>(source_file),
                                 (std::istreambuf_iterator<char>()));
                    source_file.close();

                    // Print information about compilation
                    cout << "Compiling " << _source_file_name.str() << ":"
                         << endl << parameters << endl;

                    // Create OpenCL program
                    cl::Program *program = new cl::Program(context, source);
                    try {
                        // Build the program
                        (*program).build(devices, parameters.c_str());
                        programs.push_back(program);

                    } catch (cl::Error &error) {
                        if (strcmp(error.what(), "clBuildProgram") == 0) {
                            // Print error message
                            std::string msg;
                            (*program).getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
                            std::cerr << msg << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                } // for each library

                // Fill which_program structure
                which_program[name_gridder] = 0;
                which_program[name_degridder] = 1;
                which_program[name_adder] = 2;
                which_program[name_splitter] = 3;
                which_program[name_scaler] = 4;
            } // compile

            void OpenCL::parameter_sanity_check()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }
        } // namespace opencl
    } // namespace proxy
} // namespace idg
