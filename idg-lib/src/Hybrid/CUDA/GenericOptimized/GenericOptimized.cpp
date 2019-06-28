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
            const unsigned max_nr_streams = 2;

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

                cu::Marker marker("initialize");
                marker.start();

                // Initialize devices
                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);

                    // Set device report
                    device.set_context();
                    device.set_report(report);

                    // Allocate device memory
                    cu::DeviceMemory& d_spheroidal     = device.get_device_spheroidal(subgrid_size);
                    cu::DeviceMemory& d_aterms         = device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);
                    cu::DeviceMemory& d_aterms_indices = device.get_device_aterms_indices(nr_baselines, nr_timesteps);
                    device.get_device_wavenumbers(nr_channels);

                    unsigned int avg_aterm_correction_subgrid_size = m_avg_aterm_correction.size() ? subgrid_size : 0;
                    cu::DeviceMemory& d_avg_aterm_correction = device.get_device_avg_aterm_correction(avg_aterm_correction_subgrid_size);

                    // Copy static data structures
                    cu::Stream& htodstream = device.get_htod_stream();
                    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                    htodstream.memcpyHtoDAsync(d_aterms, aterms.data());
                    htodstream.memcpyHtoDAsync(d_aterms_indices, plan.get_aterm_indices_ptr());
                    // wavenumber can differ for individual gridding/degridding calls,
                    // need to copy them in the routines rather than here

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

                marker.end();

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
                InstanceCUDA& device = get_device(0);
                device.set_context();

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

                // Page-lock host memory
                #if defined(REGISTER_HOST_MEMORY)
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

                // Load memory objects
                cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                cu::DeviceMemory& d_aterms_indices = device.get_device_aterms_indices();
                cu::DeviceMemory& d_avg_aterm_correction = device.get_device_avg_aterm_correction();

                // Load streams
                cu::Stream& executestream = device.get_execute_stream();
                cu::Stream& htodstream    = device.get_htod_stream();
                cu::Stream& dtohstream    = device.get_dtoh_stream();

                // Copy static data structures
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());

                // Id for double-buffering
                unsigned local_id = 0;

                // Iterate all jobs
                for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {

                    unsigned job_id_next = job_id + 1;
                    unsigned local_id_next = (local_id + 1) % 2;

                    // Get parameters for current iteration
                    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
                    auto current_nr_subgrids  = jobs[job_id].current_nr_subgrids;
                    void *metadata_ptr        = jobs[job_id].metadata_ptr;
                    void *uvw_ptr             = jobs[job_id].uvw_ptr;
                    void *visibilities_ptr    = jobs[job_id].visibilities_ptr;

                    // Load memory objects
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id);
                    cu::HostMemory&   h_subgrids     = device.get_host_subgrids(local_id);

                    // Copy input data for first job to device
                    if (job_id == 0) {
                        auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                        auto sizeof_uvw          = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                        auto sizeof_metadata     = auxiliary::sizeof_metadata(current_nr_subgrids);
                        htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                        htodstream.record(*inputCopied[job_id]);
                    }

                    // Wait for input to be copied
                    executestream.waitEvent(*inputCopied[job_id]);

                    // Launch gridder kernel
                    device.launch_gridder(
                        current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                        d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                        d_aterms, d_aterms_indices, d_avg_aterm_correction, d_metadata, d_subgrids);

                    // Launch FFT
                    device.launch_fft(d_subgrids, FourierDomainToImageDomain);

                    // Launch scaler
                    device.launch_scaler(current_nr_subgrids, subgrid_size, d_subgrids);
                    executestream.record(*gpuFinished[job_id]);

                    // Copy input data for next job
                    if (job_id_next < jobs.size()) {

                        // Load memory objects
                        cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id_next);
                        cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id_next);
                        cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id_next);

                        auto current_nr_baselines = jobs[job_id_next].current_nr_baselines;
                        auto current_nr_subgrids  = jobs[job_id_next].current_nr_subgrids;
                        void *metadata_ptr        = jobs[job_id_next].metadata_ptr;
                        void *uvw_ptr             = jobs[job_id_next].uvw_ptr;
                        void *visibilities_ptr    = jobs[job_id_next].visibilities_ptr;

                        // Copy input data to device
                        auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                        auto sizeof_uvw          = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                        auto sizeof_metadata     = auxiliary::sizeof_metadata(current_nr_subgrids);
                        htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                        htodstream.record(*inputCopied[job_id_next]);
                    }

                    // Copy subgrid to host
                    dtohstream.waitEvent(*gpuFinished[job_id]);
                    auto sizeof_subgrids = auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
                    dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids, sizeof_subgrids);
                    dtohstream.record(*outputCopied[job_id]);

                    // Wait for subgrid to be copied
                    device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps, jobs[job_id].current_nr_subgrids);
                    outputCopied[job_id]->synchronize();

                    // Run adder on host
                    cu::Marker marker("run_adder_wstack");
                    marker.start();
                    cpuKernels.run_adder_wstack(
                        current_nr_subgrids, grid_size, subgrid_size,
                        metadata_ptr, h_subgrids, grid.data());
                    marker.end();

                    // Update local id
                    local_id = local_id_next;
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
                InstanceCUDA& device = get_device(0);
                device.set_context();

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

                // Page-lock host memory
                #if defined(REGISTER_HOST_MEMORY)
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

                // Load memory objects
                cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                cu::DeviceMemory& d_aterms_indices = device.get_device_aterms_indices();

                // Load streams
                cu::Stream& executestream = device.get_execute_stream();
                cu::Stream& htodstream    = device.get_htod_stream();
                cu::Stream& dtohstream    = device.get_dtoh_stream();

                // Copy static data structures
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());

                // Id for double-buffering
                unsigned local_id = 0;

                // Iterate all jobs
                for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {

                    unsigned job_id_next = job_id + 1;
                    unsigned local_id_next = (local_id + 1) % 2;

                    // Get parameters for current iteration
                    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
                    auto current_nr_subgrids  = jobs[job_id].current_nr_subgrids;
                    void *metadata_ptr        = jobs[job_id].metadata_ptr;
                    void *uvw_ptr             = jobs[job_id].uvw_ptr;
                    void *visibilities_ptr    = jobs[job_id].visibilities_ptr;

                    // Load memory objects
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id);
                    cu::HostMemory&   h_subgrids     = device.get_host_subgrids(local_id);
                    #if !defined(REGISTER_HOST_MEMORY)
                    cu::HostMemory&   h_visibilities = device.get_host_visibilities(local_id);
                    #endif

                    // Copy input data for first job to device
                    if (job_id == 0) {
                        auto sizeof_uvw         = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                        auto sizeof_metadata    = auxiliary::sizeof_metadata(current_nr_subgrids);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                    }

                    // Run splitter on host
                    cu::Marker marker("run_splitter_wstack");
                    marker.start();
                    cpuKernels.run_splitter_wstack(
                        current_nr_subgrids, grid_size, subgrid_size,
                        metadata_ptr, h_subgrids, grid.data());
                    marker.end();

                    // Copy subgrids to device
                    auto sizeof_subgrids    = auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
                    htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids, sizeof_subgrids);
                    htodstream.record(*inputCopied[job_id]);

                    // Wait for input to be copied
                    executestream.waitEvent(*inputCopied[job_id]);

                    // Wait for output buffer to be free
                    if (job_id > 2) {
                        executestream.waitEvent(*outputCopied[job_id - 2]);
                    }

                    // Launch FFT
                    device.launch_fft(d_subgrids, ImageDomainToFourierDomain);

                    // Launch degridder kernel
                    device.launch_degridder(
                        current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                        d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                        d_aterms, d_aterms_indices, d_metadata, d_subgrids);
                    executestream.record(*gpuFinished[job_id]);

                    // Copy input data for next job
                    if (job_id_next < jobs.size()) {

                        // Load memory objects
                        cu::DeviceMemory& d_uvw      = device.get_device_uvw(local_id_next);
                        cu::DeviceMemory& d_metadata = device.get_device_metadata(local_id_next);

                        void *metadata_ptr        = jobs[job_id_next].metadata_ptr;
                        void *uvw_ptr             = jobs[job_id_next].uvw_ptr;

                        // Copy input data to device
                        auto sizeof_uvw         = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                        auto sizeof_metadata    = auxiliary::sizeof_metadata(current_nr_subgrids);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                        htodstream.record(*inputCopied[job_id_next]);
                    }

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

                    // Wait for degridder to finish
                    gpuFinished[job_id]->synchronize();
                    device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps, jobs[job_id].current_nr_subgrids);

                    // Update local id
                    local_id = local_id_next;
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

            void GenericOptimized::do_calibrate_init(
                std::vector<std::unique_ptr<Plan>> &&plans,
                float w_step,
                Array1D<float> &&shift,
                float cell_size,
                unsigned int kernel_size,
                unsigned int subgrid_size,
                const Array1D<float> &frequencies,
                Array4D<Visibility<std::complex<float>>> &&visibilities,
                Array4D<Visibility<float>> &&weights,
                Array3D<UVWCoordinate<float>> &&uvw,
                Array2D<std::pair<unsigned int,unsigned int>> &&baselines,
                const Grid& grid,
                const Array2D<float>& spheroidal)
            {
                InstanceCPU& cpuKernels = cpuProxy->get_kernels();
                cpuKernels.set_report(report);

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Arguments
                auto nr_antennas  = plans.size();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;
                auto nr_channels  = frequencies.get_x_dim();
                auto max_nr_terms = m_calibrate_max_nr_terms;
                auto nr_correlations = 4;

                // Allocate subgrids for all antennas
                std::vector<Array4D<std::complex<float>>> subgrids;
                subgrids.reserve(nr_antennas);

                // Start performance measurement
                #if defined(REPORT_TOTAL)
                report.initialize();
                powersensor::State states[2];
                states[0] = hostPowerSensor->read();
                #endif

                // Load device
                InstanceCUDA& device = get_device(0);
                device.set_context();
                device.free_device_memory();
                device.set_report(report);

                // Load stream
                cu::Stream& htodstream = device.get_htod_stream();

                // Maximum number of subgrids for any antenna
                unsigned int max_nr_subgrids = 0;

                // Maximum number of timesteps for any antenna
                unsigned int max_nr_timesteps = 0;

                // Reset vectors in calibration state
                m_calibrate_state.d_metadata_ids.clear();
                m_calibrate_state.d_subgrids_ids.clear();
                m_calibrate_state.d_visibilities_ids.clear();
                m_calibrate_state.d_weights_ids.clear();
                m_calibrate_state.d_uvw_ids.clear();

                // Create subgrids for every antenna
                for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++)
                {
                    // Allocate subgrids for current antenna
                    unsigned int nr_subgrids = plans[antenna_nr]->get_nr_subgrids();
                    Array4D<std::complex<float>> subgrids_(nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    if (nr_subgrids > max_nr_subgrids) {
                        max_nr_subgrids = nr_subgrids;
                    }

                    unsigned int nr_timesteps = plans[antenna_nr]->get_max_nr_timesteps_subgrid();
                    if (nr_timesteps > max_nr_timesteps) {
                        max_nr_timesteps = nr_timesteps;
                    }

                    // Get data pointers
                    void *metadata_ptr     = (void *) plans[antenna_nr]->get_metadata_ptr();
                    void *subgrids_ptr     = subgrids_.data();
                    void *grid_ptr         = grid.data();

                    // Splitter kernel
                    if (w_step == 0.0) {
                        cpuKernels.run_splitter(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    } else if (plans[antenna_nr]->get_use_wtiles()) {
                        WTileUpdateSet wtile_initialize_set = plans[antenna_nr]->get_wtile_initialize_set();
                        WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
                        cpuKernels.run_splitter_wtiles_from_grid(
                            grid_size,
                            subgrid_size,
                            image_size,
                            w_step,
                            wtile_initialize_info.wtile_ids.size(),
                            wtile_initialize_info.wtile_ids.data(),
                            wtile_initialize_info.wtile_coordinates.data(),
                            cpuProxy->getWTilesBuffer(),
                            grid.data());
                        for(unsigned int subgrid_index = 0; subgrid_index < nr_subgrids; )
                        {
                            if ((unsigned int)wtile_initialize_set.front().subgrid_index == subgrid_index)
                            {
                                wtile_initialize_set.pop_front();
                                WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
                                cpuKernels.run_splitter_wtiles_from_grid(
                                    grid_size,
                                    subgrid_size,
                                    image_size,
                                    w_step,
                                    wtile_initialize_info.wtile_ids.size(),
                                    wtile_initialize_info.wtile_ids.data(),
                                    wtile_initialize_info.wtile_coordinates.data(),
                                    cpuProxy->getWTilesBuffer(),
                                    grid_ptr);
                            }

                            unsigned int nr_subgrids_ = nr_subgrids - subgrid_index;
                            if (wtile_initialize_set.front().subgrid_index - subgrid_index < nr_subgrids_)
                            {
                                nr_subgrids_ = wtile_initialize_set.front().subgrid_index - subgrid_index;
                            }

                            cpuKernels.run_splitter_subgrids_from_wtiles(
                                nr_subgrids_,
                                grid_size,
                                subgrid_size,
                                &static_cast<Metadata*>(metadata_ptr)[subgrid_index],
                                &static_cast<std::complex<float>*>(subgrids_ptr)[subgrid_index * subgrid_size * subgrid_size * NR_CORRELATIONS],
                                cpuProxy->getWTilesBuffer());

                            subgrid_index += nr_subgrids_;
                        }
                    } else {
                        cpuKernels.run_splitter_wstack(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    }

                    // FFT kernel
                    cpuKernels.run_subgrid_fft(grid_size, subgrid_size, nr_subgrids, subgrids_ptr, CUFFT_FORWARD);

                    // Apply spheroidal
                    for (unsigned int i = 0; i < nr_subgrids; i++) {
                        for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
                            for (unsigned int j = 0; j < subgrid_size; j++) {
                                for (unsigned int k = 0; k < subgrid_size; k++) {
                                    unsigned int y = (j + (subgrid_size/2)) % subgrid_size;
                                    unsigned int x = (k + (subgrid_size/2)) % subgrid_size;
                                    subgrids_(i, pol, y, x) *= spheroidal(j,k);
                                }
                            }
                        }
                    }

                    // Allocate and initialize device memory for current antenna
                    void *visibilities_ptr   = visibilities.data(antenna_nr);
                    void *weights_ptr        = weights.data(antenna_nr);
                    void *uvw_ptr            = uvw.data(antenna_nr);
                    auto sizeof_metadata     = auxiliary::sizeof_metadata(nr_subgrids);
                    auto sizeof_subgrids     = auxiliary::sizeof_subgrids(nr_subgrids, subgrid_size);
                    auto sizeof_visibilities = auxiliary::sizeof_visibilities(nr_subgrids, nr_timesteps, nr_channels);
                    auto sizeof_weights      = sizeof_visibilities / 2;
                    auto sizeof_uvw          = auxiliary::sizeof_uvw(nr_antennas-1, nr_timesteps);
                    auto d_metadata_id       = device.allocate_device_memory(sizeof_metadata);
                    auto d_subgrids_id       = device.allocate_device_memory(sizeof_subgrids);
                    auto d_visibilities_id   = device.allocate_device_memory(sizeof_visibilities);
                    auto d_weights_id        = device.allocate_device_memory(sizeof_weights);
                    auto d_uvw_id            = device.allocate_device_memory(sizeof_uvw);
                    m_calibrate_state.d_metadata_ids.push_back(d_metadata_id);
                    m_calibrate_state.d_subgrids_ids.push_back(d_subgrids_id);
                    m_calibrate_state.d_visibilities_ids.push_back(d_visibilities_id);
                    m_calibrate_state.d_weights_ids.push_back(d_weights_id);
                    m_calibrate_state.d_uvw_ids.push_back(d_uvw_id);
                    cu::DeviceMemory& d_metadata     = device.retrieve_device_memory(d_metadata_id);
                    cu::DeviceMemory& d_subgrids     = device.retrieve_device_memory(d_subgrids_id);
                    cu::DeviceMemory& d_visibilities = device.retrieve_device_memory(d_visibilities_id);
                    cu::DeviceMemory& d_weights      = device.retrieve_device_memory(d_weights_id);
                    cu::DeviceMemory& d_uvw          = device.retrieve_device_memory(d_uvw_id);
                    htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                    htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, sizeof_subgrids);
                    htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
                    htodstream.memcpyHtoDAsync(d_weights, weights_ptr, sizeof_weights);
                    htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                    htodstream.synchronize();
                } // end for antennas

                // End performance measurement
                #if defined(REPORT_TOTAL)
                states[1] = hostPowerSensor->read();
                report.update_host(states[0], states[1]);
                report.print_total(0, 0);
                #endif

                // Set calibration state member variables
                m_calibrate_state.plans        = std::move(plans);
                m_calibrate_state.w_step       = w_step;
                m_calibrate_state.shift        = std::move(shift);
                m_calibrate_state.cell_size    = cell_size;
                m_calibrate_state.image_size   = image_size;
                m_calibrate_state.kernel_size  = kernel_size;
                m_calibrate_state.grid_size    = grid_size;
                m_calibrate_state.subgrid_size = subgrid_size;
                m_calibrate_state.nr_channels  = nr_channels;

                // Initialize wavenumbers
                cu::DeviceMemory& d_wavenumbers = device.get_device_wavenumbers(nr_channels);
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());

                // Allocate scratch device memory
                auto sizeof_scratch_sum = max_nr_subgrids * max_nr_timesteps * nr_channels * nr_correlations * max_nr_terms * sizeof(std::complex<float>);
                m_calibrate_state.d_scratch_sum_id  = device.allocate_device_memory(sizeof_scratch_sum);
            }

            void GenericOptimized::do_calibrate_update(
                const int antenna_nr,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<Matrix2x2<std::complex<float>>>& aterm_derivatives,
                Array3D<std::complex<float>>& hessian,
                Array2D<std::complex<float>>& gradient)
            {
                // Arguments
                auto nr_subgrids  = m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
                auto nr_timesteps = m_calibrate_state.plans[antenna_nr]->get_nr_timesteps();
                auto nr_channels  = m_calibrate_state.nr_channels;
                auto nr_terms     = aterm_derivatives.get_z_dim();
                auto subgrid_size = aterms.get_y_dim();
                auto nr_timeslots = aterms.get_w_dim();
                auto nr_stations  = aterms.get_z_dim();
                auto grid_size    = m_calibrate_state.grid_size;
                auto image_size   = m_calibrate_state.image_size;
                auto w_step       = m_calibrate_state.w_step;
                auto max_nr_terms = m_calibrate_max_nr_terms;

                assert((nr_terms+1) < max_nr_terms);

                // Performance measurement
                if (antenna_nr == 0) {
                    report.initialize(nr_channels, subgrid_size, 0, nr_terms);
                }

                // Data pointers
                void *aterm_ptr            = aterms.data();
                void *aterm_derivative_ptr = aterm_derivatives.data();
                void *hessian_ptr          = hessian.data();
                void *gradient_ptr         = gradient.data();

                // Start marker
                cu::Marker marker("do_calibrate_update");
                marker.start();

                // Load device
                InstanceCUDA& device = get_device(0);
                device.set_context();

                // Load streams
                cu::Stream& executestream = device.get_execute_stream();
                cu::Stream& htodstream    = device.get_htod_stream();
                cu::Stream& dtohstream    = device.get_dtoh_stream();

                // Load memory objects
                cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                cu::DeviceMemory& d_aterms       = device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);
                unsigned int d_metadata_id       = m_calibrate_state.d_metadata_ids[antenna_nr];
                unsigned int d_subgrids_id       = m_calibrate_state.d_subgrids_ids[antenna_nr];
                unsigned int d_visibilities_id   = m_calibrate_state.d_visibilities_ids[antenna_nr];
                unsigned int d_weights_id        = m_calibrate_state.d_weights_ids[antenna_nr];
                unsigned int d_uvw_id            = m_calibrate_state.d_uvw_ids[antenna_nr];
                cu::DeviceMemory& d_metadata     = device.retrieve_device_memory(d_metadata_id);
                cu::DeviceMemory& d_subgrids     = device.retrieve_device_memory(d_subgrids_id);
                cu::DeviceMemory& d_visibilities = device.retrieve_device_memory(d_visibilities_id);
                cu::DeviceMemory& d_weights      = device.retrieve_device_memory(d_weights_id);
                cu::DeviceMemory& d_uvw          = device.retrieve_device_memory(d_uvw_id);
                cu::DeviceMemory& d_scratch_sum  = device.retrieve_device_memory(m_calibrate_state.d_scratch_sum_id);

                // Allocate additional data structures
                cu::DeviceMemory d_aterms_deriv(aterm_derivatives.bytes());
                cu::DeviceMemory d_hessian(hessian.bytes());
                cu::DeviceMemory d_gradient(gradient.bytes());
                cu::HostMemory h_hessian(hessian.bytes());
                cu::HostMemory h_gradient(gradient.bytes());

                // Events
                cu::Event inputCopied, executeFinished, outputCopied;

                // Copy input data to device
                htodstream.memcpyHtoDAsync(d_aterms, aterm_ptr);
                htodstream.memcpyHtoDAsync(d_aterms_deriv, aterm_derivative_ptr);
                htodstream.memcpyHtoDAsync(d_hessian, hessian_ptr);
                htodstream.memcpyHtoDAsync(d_gradient, gradient_ptr);
                htodstream.record(inputCopied);

                // Get max number of timesteps for any subgrid
                auto max_nr_timesteps = m_calibrate_state.plans[antenna_nr]->get_max_nr_timesteps_subgrid();

                // Run calibration update step
                executestream.waitEvent(inputCopied);
                device.launch_calibrate(
                    nr_subgrids, grid_size, subgrid_size, image_size, w_step, max_nr_timesteps, nr_channels, nr_terms,
                    d_uvw, d_wavenumbers, d_visibilities, d_weights, d_aterms, d_aterms_deriv, d_metadata, d_subgrids,
                    d_scratch_sum, d_hessian, d_gradient);
                executestream.record(executeFinished);

                // Copy output to host
                dtohstream.waitEvent(executeFinished);
                dtohstream.memcpyDtoHAsync(h_hessian, d_hessian);
                dtohstream.memcpyDtoHAsync(h_gradient, d_gradient);
                dtohstream.record(outputCopied);

                // Wait for output to finish
                outputCopied.synchronize();

                // Copy output on host
                memcpy(hessian_ptr, h_hessian, hessian.bytes());
                memcpy(gradient_ptr, h_gradient, gradient.bytes());

                // End marker
                marker.end();

                // Performance reporting
                auto nr_visibilities = nr_timesteps * nr_channels;
                report.update_total(nr_subgrids, nr_timesteps, nr_visibilities);
            }

            void GenericOptimized::do_calibrate_finish()
            {
                // Performance reporting
                #if defined(REPORT_TOTAL)
                auto nr_antennas  = m_calibrate_state.plans.size();
                auto total_nr_timesteps = 0;
                auto total_nr_subgrids  = 0;
                for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
                    total_nr_timesteps += m_calibrate_state.plans[antenna_nr]->get_nr_timesteps();
                    total_nr_subgrids  += m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
                }
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                report.print_visibilities(auxiliary::name_calibrate);
                #endif
            }

            Plan* GenericOptimized::make_plan(
                const int kernel_size,
                const int subgrid_size,
                const int grid_size,
                const float cell_size,
                const Array1D<float>& frequencies,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array1D<unsigned int>& aterms_offsets,
                Plan::Options options)
            {
                // Defer call to cpuProxy
                // cpuProxy manages the wtiles state
                // plan will be made accordingly
                return cpuProxy->make_plan(
                kernel_size,
                subgrid_size,
                grid_size,
                cell_size,
                frequencies,
                uvw,
                baselines,
                aterms_offsets,
                options);
            }



        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#include "GenericOptimizedC.h"
