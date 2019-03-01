#include "GenericOptimized.h"

#include <cuda.h>
#include <cudaProfiler.h>

#include <algorithm> // max_element
#include <mutex>

#include "InstanceCUDA.h"

/*
 * Option to enable/disable the _wwstack
 * version of the adder and splitter kernels.
 */
#define ENABLE_WSTACKING 1

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;


namespace idg {
    namespace proxy {
        namespace hybrid {

            // The maximum number of CUDA streams in any routine
            const unsigned max_nr_streams = 3;

            // Constructor
            GenericOptimized::GenericOptimized() :
                CUDA(default_info())
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Initialize cpu proxy
                cpuProxy = new idg::proxy::cpu::Optimized();

                // Initialize host PowerSensor
                hostPowerSensor = get_power_sensor(sensor_host);

                // Initialize host stream
                hostStream = new cu::Stream();

                omp_set_nested(true);

                cuProfilerStart();
            }

            // Destructor
            GenericOptimized::~GenericOptimized() {
                delete cpuProxy;
                delete hostPowerSensor;
                delete hostStream;
                cuProfilerStop();
            }

            /*
             * FFT
             */
            void GenericOptimized::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                std::cout << "Transform direction: " << direction << std::endl;
                #endif

                cpuProxy->transform(direction, grid);
            } // end transform


            /*
             * Gridding
             */
            void GenericOptimized::synchronize() {
                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    device.get_htod_stream().synchronize();
                    device.get_execute_stream().synchronize();
                    device.get_dtoh_stream().synchronize();
                }

                hostStream->synchronize();
            }

            void GenericOptimized::initialize(
                const Plan& plan,
                const float w_step,
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size,
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw std::invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                synchronize();

                // Arguments
                auto nr_channels  = frequencies.get_x_dim();
                auto nr_stations  = aterms.get_z_dim();
                auto nr_timeslots = aterms.get_w_dim();
                auto grid_size    = grid.get_x_dim();
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();

                // Initialize report
                report.initialize(nr_channels, subgrid_size, grid_size);

                // Initialize devices
                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);

                    // Set device report
                    device.set_context();
                    device.set_report(report);

                    // Set device memory
                    cu::Stream&       htodstream   = device.get_htod_stream();
                    cu::DeviceMemory& d_spheroidal = device.get_device_spheroidal(subgrid_size);

                    unsigned int avg_aterm_correction_subgrid_size = m_avg_aterm_correction.size() ? subgrid_size : 0;
                    cu::DeviceMemory& d_avg_aterm_correction = device.get_device_avg_aterm_correction(avg_aterm_correction_subgrid_size);

                    cu::DeviceMemory& d_aterms     = device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);

                    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                    htodstream.memcpyHtoDAsync(d_aterms, aterms.data());

                    if (avg_aterm_correction_subgrid_size)
                    {
                        htodstream.memcpyHtoDAsync(d_avg_aterm_correction, m_avg_aterm_correction.data());
                    }
                }

                // Set host report
                cpuProxy->get_kernels().set_report(report);

                jobsize_ = compute_jobsize(plan, nr_stations, nr_timeslots, nr_timesteps, nr_channels, subgrid_size, max_nr_streams);

                // Initialize memory/fft
                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device  = get_device(d);

                    // Compute maximum number of subgrids for this plan
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize_[d]);
                    if (planned_max_nr_subgrids.size() <= d) {
                        planned_max_nr_subgrids.push_back(0);
                    }
                    planned_max_nr_subgrids[d] = max_nr_subgrids;

                    // Initialize memory
                    for (unsigned t = 0; t < max_nr_streams; t++) {
                        device.get_device_wavenumbers(t, nr_channels);
                        device.get_device_visibilities(t, jobsize_[d], nr_timesteps, nr_channels);
                        device.get_device_uvw(t, jobsize_[d], nr_timesteps);
                        device.get_device_subgrids(t, max_nr_subgrids, subgrid_size);
                        device.get_device_metadata(t, max_nr_subgrids);
                        device.get_host_subgrids(t, max_nr_subgrids, subgrid_size);
                        device.get_host_metadata(t, max_nr_subgrids);
                        #if !defined(REGISTER_HOST_MEMORY)
                        device.get_host_visibilities(t, jobsize_[d], nr_timesteps, nr_channels);
                        #endif
                    }

                    // Plan subgrid fft
                    device.plan_fft(subgrid_size, max_nr_subgrids);
                }

                // Host power measurement
                hostStartState = hostPowerSensor->read();
            } // end initialize

            void GenericOptimized::finish(
                std::string name)
            {
                synchronize();

                State hostEndState = hostPowerSensor->read();
                report.update_host(hostStartState, hostEndState);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                report.print_total();
                report.print_devices();
                report.print_visibilities(name);
                std::clog << std::endl;
                #endif
                report.reset();
                planned_max_nr_subgrids.clear();

                for (unsigned d = 0; d < get_num_devices(); d++) {
                    get_device(d).free_device_memory();
                    get_device(d).free_host_memory();
                    get_device(d).free_fft_plans();
                }
            } // end finish

            typedef struct {
                void *dst;
                void *src;
                size_t bytes;
            } MemData;

            void copy_memory(CUstream, CUresult, void *userData)
            {
                MemData *data = static_cast<MemData*>(userData);
                char message[80];
                snprintf(message, 80, "memcpy(%p, %p, %zu)", data->dst, data->src, data->bytes);
                cu::Marker marker(message, 0xffff0000);
                marker.start();
                memcpy(data->dst, data->src, data->bytes);
                marker.end();
                delete data;
            }

            void enqueue_copy(
                cu::Stream& stream,
                void *dst,
                void *src,
                size_t bytes)
            {
                // Fill MemData struct
                MemData *data = new MemData();
                data->dst     = dst;
                data->src     = src;
                data->bytes   = bytes;

                // Enqueue memory copy
                stream.addCallback((CUstreamCallback) &copy_memory, data);
            } // end enqueue_copy

            typedef struct {
                Report *report;
                std::vector<InstanceCUDA*> devices;
                std::vector<State> startStates;
                std::vector<State> endStates;
            } StateData;

            void start_device_measurement(CUstream, CUresult, void *userData)
            {
                StateData *data = static_cast<StateData*>(userData);

                // Start device measurement
                for (unsigned d = 0; d < data->devices.size(); d++) {
                    data->startStates.push_back(data->devices[d]->measure());
                }
            }

            void end_device_measurement(CUstream, CUresult, void *userData)
            {
                StateData *data = static_cast<StateData*>(userData);

                // End device measurement
                for (unsigned d = 0; d < data->devices.size(); d++) {
                    data->endStates.push_back(data->devices[d]->measure());
                }

                // Update device report
                data->report->update_devices(data->startStates, data->endStates);

                // Cleanup
                delete data;
            }

            void GenericOptimized::run_gridding(
                const Plan& plan,
                const float w_step,
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size,
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                InstanceCPU& cpuKernels = cpuProxy->get_kernels();

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto image_size      = cell_size * grid_size;

                // Configuration
                const unsigned nr_devices = get_num_devices();
                const unsigned nr_streams = 3;

                // Page-lock host memory
                #if defined(REGISTER_HOST_MEMORY)
                InstanceCUDA& device = get_device(0);
                device.get_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities.data());
                device.get_host_uvw(nr_baselines, nr_timesteps, uvw.data());
                #endif

                // Reduce jobsize when the maximum number of subgrids for the current plan exceeds the planned number
                for (unsigned d = 0; d < nr_devices; d++) {
                    while (planned_max_nr_subgrids[d] < plan.get_max_nr_subgrids(0, nr_baselines, jobsize_[d])) {
                        jobsize_[d] *= 0.9;
                    }
                }

                // Performance measurements
                StateData *stateData = new StateData();
                stateData->report = (Report *) &report;
                for (unsigned d = 0; d < nr_devices; d++) {
                    stateData->devices.push_back(&get_device(d));
                }

                // Enqueue start device measurement
                hostStream->addCallback((CUstreamCallback) &start_device_measurement, stateData);

                int jobsize = jobsize_[0];

                // Events
                std::vector<cu::Event*> inputCopied;
                std::vector<cu::Event*> gpuFinished;
                std::vector<cu::Event*> outputCopied;

                // Prepare job data
                device.set_context();
                struct JobData {
                    unsigned current_nr_baselines;
                    unsigned current_nr_subgrids;
                    unsigned current_nr_timesteps;
                    void *metadata_ptr;
                    void *uvw_ptr;
                    void *visibilities_ptr;
                };

                std::vector<JobData> jobs;
                for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
                    unsigned int first_bl, last_bl, current_nr_baselines;
                    plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                    if (current_nr_baselines == 0) continue;
                    JobData job;
                    job.current_nr_baselines = current_nr_baselines;
                    job.current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                    job.current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                    job.metadata_ptr         = (void *) plan.get_metadata_ptr(first_bl);
                    job.uvw_ptr              = uvw.data(first_bl, 0);
                    job.visibilities_ptr     = visibilities.data(first_bl, 0, 0);
                    jobs.push_back(job);
                    inputCopied.push_back(new cu::Event());
                    gpuFinished.push_back(new cu::Event());
                    outputCopied.push_back(new cu::Event());
                }

                // Iterate all jobs
                #pragma omp parallel for ordered schedule(static,1) num_threads(nr_streams)
                for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
                    unsigned local_id = omp_get_thread_num();

                    // Get parameters for current iteration
                    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
                    auto current_nr_subgrids  = jobs[job_id].current_nr_subgrids;
                    void *metadata_ptr        = jobs[job_id].metadata_ptr;
                    void *uvw_ptr             = jobs[job_id].uvw_ptr;
                    void *visibilities_ptr    = jobs[job_id].visibilities_ptr;

                    // Load device
                    InstanceCUDA& device  = get_device(0);
                    device.set_context();

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                    cu::DeviceMemory& d_avg_aterm_correction = device.get_device_avg_aterm_correction();
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id);
                    cu::HostMemory&   h_subgrids     = device.get_host_subgrids(local_id);

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    #pragma omp critical (gridderlock)
                    {
                        // Copy input data to device
                        if (job_id < nr_streams) {
                            auto sizeof_wavenumbers  = auxiliary::sizeof_wavenumbers(nr_channels);
                            auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                            auto sizeof_uvw          = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                            auto sizeof_metadata     = auxiliary::sizeof_metadata(current_nr_subgrids);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                            htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(), sizeof_wavenumbers);
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                            htodstream.record(*inputCopied[job_id]);
                        }
                        executestream.waitEvent(*inputCopied[job_id]);

                        // Launch gridder kernel
                        device.launch_gridder(
                            current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                            d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_avg_aterm_correction, d_metadata, d_subgrids);

                        // Launch gridder post-processing kernel
                        device.launch_gridder_post(
                            current_nr_subgrids, subgrid_size, nr_stations,
                            d_spheroidal, d_aterms, d_avg_aterm_correction, d_metadata, d_subgrids);

                        // Launch FFT
                        device.launch_fft(d_subgrids, FourierDomainToImageDomain);

                        // Launch scaler
                        device.launch_scaler(current_nr_subgrids, subgrid_size, d_subgrids);
                        executestream.record(*gpuFinished[job_id]);

                        // Copy subgrid to host
                        dtohstream.waitEvent(*gpuFinished[job_id]);
                        auto sizeof_subgrids = auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
                        dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids, sizeof_subgrids);
                        dtohstream.record(*outputCopied[job_id]);
                    }

                    // Copy input data for next job
                    unsigned job_id_next = job_id + nr_streams;
                    if (job_id_next < jobs.size()) {
                        // Wait for input from other threads to be copied
                        htodstream.synchronize();

                        // Wait for computation to be finished
                        gpuFinished[job_id]->synchronize();

                        auto current_nr_baselines = jobs[job_id_next].current_nr_baselines;
                        auto current_nr_subgrids  = jobs[job_id_next].current_nr_subgrids;
                        void *metadata_ptr        = jobs[job_id_next].metadata_ptr;
                        void *uvw_ptr             = jobs[job_id_next].uvw_ptr;
                        void *visibilities_ptr    = jobs[job_id_next].visibilities_ptr;

                        // Copy input data to device
                        auto sizeof_wavenumbers  = auxiliary::sizeof_wavenumbers(nr_channels);
                        auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                        auto sizeof_uvw          = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                        auto sizeof_metadata     = auxiliary::sizeof_metadata(current_nr_subgrids);
                        htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(), sizeof_wavenumbers);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                        htodstream.record(*inputCopied[job_id_next]);
                    }

                    // Wait for subgrid to be copied
                    device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps, jobs[job_id].current_nr_subgrids);
                    outputCopied[job_id]->synchronize();

                    // Run adder on CPU
                    #pragma omp critical (adderlock)
                    {
                        cu::Marker marker("run_adder_wstack");
                        marker.start();
                        cpuKernels.run_adder_wstack(
                            current_nr_subgrids, grid_size, subgrid_size,
                            metadata_ptr, h_subgrids, grid.data());
                        marker.end();
                    }
                } // end for bl

                // Enqueue end device measurement
                hostStream->addCallback((CUstreamCallback) &end_device_measurement, stateData);

                // Update report
                auto total_nr_subgrids     = plan.get_nr_subgrids();
                auto total_nr_timesteps    = plan.get_nr_timesteps();
                auto total_nr_visibilities = plan.get_nr_visibilities();
                report.update_total(total_nr_subgrids, total_nr_timesteps, total_nr_visibilities);
            } // end run_gridding

            void GenericOptimized::do_gridding(
                const Plan& plan,
                const float w_step, // in lambda
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const unsigned int subgrid_size,
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
                std::cout << __func__ << std::endl;
                #endif

                std::clog << "### Initialize gridding" << std::endl;
                initialize(
                    plan,
                    w_step,
                    shift,
                    cell_size,
                    kernel_size,
                    subgrid_size,
                    frequencies,
                    visibilities,
                    uvw,
                    baselines,
                    grid,
                    aterms,
                    aterms_offsets,
                    spheroidal);

                std::clog << "### Run gridding" << std::endl;
                run_gridding(
                    plan,
                    w_step,
                    shift,
                    cell_size,
                    kernel_size,
                    subgrid_size,
                    frequencies,
                    visibilities,
                    uvw,
                    baselines,
                    grid,
                    aterms,
                    aterms_offsets,
                    spheroidal);

                std::clog << "### Finish gridding" << std::endl;
                finish_gridding();
            } // end do_gridding


            /*
             * Degridding
             */
            void GenericOptimized::run_degridding(
                const Plan& plan,
                const float w_step,
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size,
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                InstanceCPU& cpuKernels = cpuProxy->get_kernels();

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto image_size      = cell_size * grid_size;

                // Configuration
                const unsigned nr_devices = get_num_devices();
                const unsigned nr_streams = 3;

                // Page-lock host memory
                #if defined(REGISTER_HOST_MEMORY)
                InstanceCUDA& device = get_device(0);
                device.get_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities.data());
                device.get_host_uvw(nr_baselines, nr_timesteps, uvw.data());
                #endif

                // Reduce jobsize when the maximum number of subgrids for the current plan exceeds the planned number
                for (unsigned d = 0; d < nr_devices; d++) {
                    while (planned_max_nr_subgrids[d] < plan.get_max_nr_subgrids(0, nr_baselines, jobsize_[d])) {
                        jobsize_[d] *= 0.9;
                    }
                }

                // Performance measurements
                StateData *stateData = new StateData();
                stateData->report = (Report *) &report;
                for (unsigned d = 0; d < nr_devices; d++) {
                    stateData->devices.push_back(&get_device(d));
                }

                // Enqueue start device measurement
                hostStream->addCallback((CUstreamCallback) &start_device_measurement, stateData);

                int jobsize = jobsize_[0];

                // Events
                std::vector<cu::Event*> inputCopied;
                std::vector<cu::Event*> gpuFinished;
                std::vector<cu::Event*> outputCopied;

                // Prepare job data
                device.set_context();
                struct JobData {
                    unsigned current_nr_baselines;
                    unsigned current_nr_subgrids;
                    unsigned current_nr_timesteps;
                    void *metadata_ptr;
                    void *uvw_ptr;
                    void *visibilities_ptr;
                };

                std::vector<JobData> jobs;
                for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
                    unsigned int first_bl, last_bl, current_nr_baselines;
                    plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                    if (current_nr_baselines == 0) continue;
                    JobData job;
                    job.current_nr_baselines = current_nr_baselines;
                    job.current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                    job.current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                    job.metadata_ptr         = (void *) plan.get_metadata_ptr(first_bl);
                    job.uvw_ptr              = uvw.data(first_bl, 0);
                    job.visibilities_ptr     = visibilities.data(first_bl, 0, 0);
                    jobs.push_back(job);
                    inputCopied.push_back(new cu::Event());
                    gpuFinished.push_back(new cu::Event());
                    outputCopied.push_back(new cu::Event());
                }

                // Iterate all jobs
                #pragma omp parallel for ordered schedule(static,1) num_threads(nr_streams)
                for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
                    unsigned local_id = omp_get_thread_num();

                    // Get parameters for current iteration
                    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
                    auto current_nr_subgrids  = jobs[job_id].current_nr_subgrids;
                    void *metadata_ptr        = jobs[job_id].metadata_ptr;
                    void *uvw_ptr             = jobs[job_id].uvw_ptr;
                    void *visibilities_ptr    = jobs[job_id].visibilities_ptr;

                    // Load device
                    InstanceCUDA& device  = get_device(0);
                    device.set_context();

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers(local_id, 0);
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id);
                    cu::HostMemory&   h_subgrids     = device.get_host_subgrids(local_id);
                    #if !defined(REGISTER_HOST_MEMORY)
                    cu::HostMemory&   h_visibilities = device.get_host_visibilities(local_id);
                    #endif

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Run splitter on CPU
                    #pragma omp critical (splitterlock)
                    {
                        cu::Marker marker("run_splitter_wstack");
                        marker.start();
                        cpuKernels.run_splitter_wstack(
                            current_nr_subgrids, grid_size, subgrid_size,
                            metadata_ptr, h_subgrids, grid.data());
                        marker.end();
                    }

                    #pragma omp critical (degridderlock)
                    {
                        // Copy input data to device
                        auto sizeof_wavenumbers = auxiliary::sizeof_wavenumbers(nr_channels);
                        auto sizeof_uvw         = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                        auto sizeof_metadata    = auxiliary::sizeof_metadata(current_nr_subgrids);
                        auto sizeof_subgrids    = auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(), sizeof_wavenumbers);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                        htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids, sizeof_subgrids);
                        htodstream.record(*inputCopied[job_id]);

                        // Launch FFT
                        executestream.waitEvent(*inputCopied[job_id]);
                        device.launch_fft(d_subgrids, ImageDomainToFourierDomain);

                        // Launch degridder pre-processing kernel
                        device.launch_degridder_pre(
                            current_nr_subgrids, subgrid_size, nr_stations,
                            d_spheroidal, d_aterms, d_metadata, d_subgrids);

                        // Launch degridder kernel
                        device.launch_degridder(
                            current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                            d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                        executestream.record(*gpuFinished[job_id]);

                        // Copy visibilities to host
                        dtohstream.waitEvent(*gpuFinished[job_id]);
                        auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                        #if defined(REGISTER_HOST_MEMORY)
                        dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, sizeof_visibilities);
                        #else
                        dtohstream.memcpyDtoHAsync(h_visibilities, d_visibilities, sizeof_visibilities);
                        enqueue_copy(dtohstream, visibilities_ptr, h_visibilities, sizeof_visibilities);
                        #endif
                        dtohstream.record(*outputCopied[job_id]);
                    }

                    // Finish job
                    device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps, jobs[job_id].current_nr_subgrids);
                    outputCopied[job_id]->synchronize();
                } // end for bl

                // Enqueue end device measurement
                hostStream->addCallback((CUstreamCallback) &end_device_measurement, stateData);

                // Update report
                auto total_nr_subgrids     = plan.get_nr_subgrids();
                auto total_nr_timesteps    = plan.get_nr_timesteps();
                auto total_nr_visibilities = plan.get_nr_visibilities();
                report.update_total(total_nr_subgrids, total_nr_timesteps, total_nr_visibilities);
            } // end run_degridding

            void GenericOptimized::do_degridding(
                const Plan& plan,
                const float w_step, // in lambda
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const unsigned int subgrid_size,
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
                std::cout << __func__ << std::endl;
                #endif

                std::clog << "### Initialize degridding" << std::endl;
                initialize(
                    plan,
                    w_step,
                    shift,
                    cell_size,
                    kernel_size,
                    subgrid_size,
                    frequencies,
                    visibilities,
                    uvw,
                    baselines,
                    grid,
                    aterms,
                    aterms_offsets,
                    spheroidal);

                std::clog << "### Run degridding" << std::endl;
                run_degridding(
                    plan,
                    w_step,
                    shift,
                    cell_size,
                    kernel_size,
                    subgrid_size,
                    frequencies,
                    visibilities,
                    uvw,
                    baselines,
                    grid,
                    aterms,
                    aterms_offsets,
                    spheroidal);

                std::clog << "### Finish degridding" << std::endl;
                finish_degridding();
            } // end do_degridding

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#include "GenericOptimizedC.h"
