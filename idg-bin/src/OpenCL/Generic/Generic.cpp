#include "Generic.h"

using namespace std;
using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            Generic::Generic(
                Parameters params) :
                OpenCLNew(params)
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
                    #if defined(MEASURE_POWER_ARDUINO)
                    for (PerformanceCounter& counter : counters) {
                        counter.setPowerSensor(&powerSensor);
                    }
                    #endif
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
                                executequeue, current_nr_timesteps, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);

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
            }

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
                #if defined(MEASURE_POWER_ARDUINO)
                counter_fft.setPowerSensor(&powerSensor);
                #endif

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
