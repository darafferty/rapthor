#include "Generic.h"

using namespace std;
using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            Generic::Generic(
                Parameters params) :
                OpenCL(params)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif
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
				cl::Context &context      = device->get_context();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
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
                cl::Buffer h_visibilities(context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(context, CL_MEM_ALLOC_HOST_PTR, sizeof_metadata(total_nr_subgrids));

                // Copy input data to host memory
                htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0, sizeof_visibilities(nr_baselines), visibilities);
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0, sizeof_metadata(total_nr_subgrids), metadata);

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
                    #if ENABLE_WARMUP
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
                                executequeue, current_nr_timesteps, current_nr_subgrids, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);

							// Launch FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);

                            // Launch adder kernel
                            kernel_adder->launchAsync(executequeue, current_nr_subgrids, d_metadata, d_subgrids, d_grid, counters[3]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
                        }
                    }

                    outputReady[0].wait();
                }

                // Copy grid to host
                executequeue.finish();
                PowerSensor::State stopState = power_sensor->read();
                dtohqueue.enqueueReadBuffer(d_grid, CL_TRUE, 0, sizeof_grid(), grid, NULL, NULL);
                dtohqueue.finish();

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

                // Load device
                DeviceInstance *device    = devices[0];
                PowerSensor *power_sensor = device->get_powersensor();
				cl::Context &context      = device->get_context();

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
                unique_ptr<Degridder> kernel_degridder = device->get_kernel_degridder();
                unique_ptr<Splitter> kernel_splitter   = device->get_kernel_splitter();;
                unique_ptr<GridFFT> kernel_fft         = device->get_kernel_fft();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cl::CommandQueue executequeue = device->get_execute_queue();
                cl::CommandQueue htodqueue    = device->get_htod_queue();
                cl::CommandQueue dtohqueue    = device->get_dtoh_queue();
                const int nr_streams = 3;

                // Host memory
                cl::Buffer h_visibilities(context, CL_MEM_ALLOC_HOST_PTR, sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(context, CL_MEM_ALLOC_HOST_PTR, sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(context, CL_MEM_ALLOC_HOST_PTR, sizeof_metadata(total_nr_subgrids));

                // Copy input data to host memory
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0, sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0, sizeof_metadata(total_nr_subgrids), metadata);

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
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata(max_nr_subgrids));

                    // Warmup
                    #if ENABLE_WARMUP
                    htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, 0, 0, sizeof_uvw(jobsize), NULL, NULL);
                    htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, 0, 0, sizeof_metadata(max_nr_subgrids), NULL, NULL);
                    htodqueue.finish();
                    kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                    executequeue.finish();
                    #endif

                    // Events
                    vector<cl::Event> inputReady(1), computeReady(1), outputReady(1);
                    htodqueue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);

                    // Performance counters
                    vector<PerformanceCounter> counters(3);
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
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, sizeof_metadata(current_nr_subgrids), NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch splitter kernel
                            executequeue.enqueueBarrierWithWaitList(&inputReady, NULL);
                            kernel_splitter->launchAsync(executequeue, current_nr_subgrids, d_metadata, d_subgrids, d_grid, counters[0]);

        					// Launch FFT
                            kernel_fft->launchAsync(executequeue, d_subgrids, CLFFT_FORWARD);

        					// Launch degridder kernel
                            kernel_degridder->launchAsync(
                                executequeue, current_nr_timesteps, current_nr_subgrids, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[2]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

        					// Copy visibilities to host
                            dtohqueue.enqueueBarrierWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, sizeof_visibilities(current_nr_baselines), NULL, &outputReady[0]);
                        }

                        outputReady[0].wait();
                    }
                }

                // Copy visibilities
                dtohqueue.finish();
                PowerSensor::State stopState = power_sensor->read();
                dtohqueue.enqueueReadBuffer(h_visibilities, CL_TRUE, 0, sizeof_visibilities(nr_baselines), visibilities, NULL, NULL);
                dtohqueue.finish();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                uint64_t total_flops_degridder  = kernel_degridder->flops(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, total_nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, total_nr_subgrids);
                uint64_t total_flops_splitter   = kernel_splitter->flops(total_nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                double total_runtime_degridding = power_sensor->seconds(startState, stopState);
                double total_watt_degridding    = power_sensor->Watt(startState, stopState);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif

            } // end grid_visibilities

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
				cl::Context &context      = device->get_context();

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
                cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid());

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
                inputReady[0].wait();
                time_input += omp_get_wtime();

                // Create FFT plan
                kernel_fft->plan(context, queue, gridsize, 1);

				// Launch FFT
                kernel_fft->launchAsync(queue, d_grid, sign);
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
