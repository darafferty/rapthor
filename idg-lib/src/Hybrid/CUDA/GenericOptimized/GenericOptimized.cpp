#include "GenericOptimized.h"

#include <cuda.h>
#include <cudaProfiler.h>

#include <algorithm> // max_element

#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;


namespace idg {
    namespace proxy {
        namespace hybrid {

            // The maximum number of CUDA streams in any routine
            const int max_nr_streams = 3;

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

                omp_set_nested(true);

                cuProfilerStart();
            }

            // Destructor
            GenericOptimized::~GenericOptimized() {
                delete cpuProxy;
                delete hostPowerSensor;
                cuProfilerStop();
            }

            /* High level routines */
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


            void GenericOptimized::initialize_memory(
                const Plan& plan,
                const std::vector<int> jobsize,
                const int nr_streams,
                const int nr_baselines,
                const int nr_timesteps,
                const int nr_channels,
                const int nr_stations,
                const int nr_timeslots,
                const int subgrid_size,
                void *visibilities,
                void *uvw)
            {
                for (int d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    device.set_context();
                    int max_jobsize = * max_element(begin(jobsize), end(jobsize));
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, max_jobsize);

                    // Static memory
                    cu::Stream&       htodstream    = device.get_htod_stream();
                    cu::DeviceMemory& d_wavenumbers = device.get_device_wavenumbers(nr_channels);
                    cu::DeviceMemory& d_spheroidal  = device.get_device_spheroidal(subgrid_size);
                    cu::DeviceMemory& d_aterms      = device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);

                    // Dynamic memory (per thread)
                    for (int t = 0; t < nr_streams; t++) {
                        device.get_device_visibilities(t, jobsize[d], nr_timesteps, nr_channels);
                        device.get_device_uvw(t, jobsize[d], nr_timesteps);
                        device.get_device_subgrids(t, max_nr_subgrids, subgrid_size);
                        device.get_device_metadata(t, max_nr_subgrids);
                        device.get_host_subgrids(t, max_nr_subgrids, subgrid_size);
                    }

                    // Host memory
                    if (d == 0) {
                        device.get_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities);
                        device.get_host_uvw(nr_baselines, nr_timesteps, uvw);
                    }
                }
            } // end initialize_memory


            void GenericOptimized::do_gridding(
                const Plan& plan,
                const float w_step, // in lambda
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

                InstanceCPU& cpuKernels = cpuProxy->get_kernels();

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw std::invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                check_dimensions(
                    frequencies, visibilities, uvw, baselines,
                    grid, aterms, aterms_offsets, spheroidal);

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto nr_timeslots    = aterms.get_w_dim();
                auto nr_correlations = grid.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto nr_w_layers     = grid.get_z_dim();
                auto image_size      = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 3;

                // Initialize metadata
                const Metadata *metadata  = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, max_nr_streams);

                // Initialize memory
                initialize_memory(
                    plan, jobsize_, nr_streams,
                    nr_baselines, nr_timesteps, nr_channels, nr_stations, nr_timeslots, subgrid_size,
                    visibilities.data(), uvw.data());

                // Performance measurements
                report.initialize(nr_channels, subgrid_size, grid_size);
                std::vector<State> startStates(nr_devices+1);
                std::vector<State> endStates(nr_devices+1);
                startStates[nr_devices] = hostPowerSensor->read();

                // Locks
                int locks[nr_devices];

                omp_set_nested(true);
                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = jobsize_[device_id];
                    int lock      = locks[device_id];
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                    // Initialize device
                    InstanceCUDA& device  = get_device(device_id);
                    device.set_context();
                    device.set_report(report);

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id);
                    cu::HostMemory&   h_subgrids     = device.get_host_subgrids(local_id);

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();
                    cu::Stream  reportStream;

                    // Copy static data structures
                    if (local_id == 0) {
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                        htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                        htodstream.memcpyHtoDAsync(d_aterms, aterms.data());
                        htodstream.synchronize();
                    }

                    // Create FFT plan
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
                    }

                    // Events
                    cu::Event inputFree;
                    cu::Event inputReady;
                    cu::Event outputFree;
                    cu::Event outputReady;
                    cu::Event adderFinished;

                    // Power measurement
                    if (local_id == 0) {
                        startStates[device_id] = device.measure();
                    }

                    #pragma omp barrier
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        unsigned int first_bl, last_bl, current_nr_baselines;
                        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                        if (current_nr_baselines == 0) continue;

                        // Initialize iteration
                        auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        void *metadata_ptr        = (void *) plan.get_metadata_ptr(first_bl);
                        void *uvw_ptr             = uvw.data(first_bl, 0);
                        void *visibilities_ptr    = visibilities.data(first_bl, 0, 0);

                        // Power measurement
                        cpuKernels.set_report(report);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device memory
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                                auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                auxiliary::sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            executestream.waitEvent(outputFree);
                            device.launch_gridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);

                            // Launch FFT
                            device.launch_fft(d_subgrids, FourierDomainToImageDomain);

                            // Launch scaler kernel
                            device.launch_scaler(
                                current_nr_subgrids, subgrid_size, d_subgrids);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

                            // Copy subgrid to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids,
                                auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size));
                            dtohstream.record(outputFree);

                            device.enqueue_report(dtohstream, current_nr_timesteps, current_nr_subgrids);
                        }

                        outputFree.synchronize();

                        // Add subgrids to grid
                        #pragma omp critical (CPU)
                        {
                            cpuKernels.run_adder_wstack(current_nr_subgrids, grid_size, subgrid_size, nr_w_layers, metadata_ptr, h_subgrids, grid.data());
                        }

                        dtohstream.record(adderFinished);
                    } // end for bl

                    // Wait for all jobs to finish
                    executestream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        endStates[device_id] = device.measure();
                    }
                } // end omp parallel

                // End timing
                endStates[nr_devices] = hostPowerSensor->read();
                report.update_host(startStates[nr_devices], endStates[nr_devices]);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auto total_nr_subgrids        = plan.get_nr_subgrids();
                auto total_nr_timesteps       = plan.get_nr_timesteps();
                auto total_nr_visibilities    = plan.get_nr_visibilities();
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                startStates.pop_back(); endStates.pop_back();
                report.print_devices(startStates, endStates);
                report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
                std::clog << std::endl;
                #endif
            } // end gridding


            void GenericOptimized::do_degridding(
                const Plan& plan,
                const float w_step, // in lambda
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

                InstanceCPU& cpuKernels = cpuProxy->get_kernels();

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw std::invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                check_dimensions(
                    frequencies, visibilities, uvw, baselines,
                    grid, aterms, aterms_offsets, spheroidal);

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto nr_timeslots    = aterms.get_w_dim();
                auto nr_correlations = grid.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto image_size      = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 3;

                // Initialize metadata
                const Metadata *metadata  = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, max_nr_streams);

                // Initialize memory
                initialize_memory(
                    plan, jobsize_, nr_streams,
                    nr_baselines, nr_timesteps, nr_channels, nr_stations, nr_timeslots, subgrid_size,
                    visibilities.data(), uvw.data());

                // Performance measurements
                report.initialize(nr_channels, subgrid_size, grid_size);
                std::vector<State> startStates(nr_devices+1);
                std::vector<State> endStates(nr_devices+1);
                startStates[nr_devices] = hostPowerSensor->read();

                // Locks
                int locks[nr_devices];

                omp_set_nested(true);
                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = jobsize_[device_id];
                    int lock      = locks[device_id];
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                    // Initialize device
                    InstanceCUDA& device  = get_device(device_id);
                    device.set_context();

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id);
                    cu::HostMemory&   h_subgrids     = device.get_host_subgrids(local_id);

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Copy static data structures
                    if (local_id == 0) {
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                        htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                        htodstream.memcpyHtoDAsync(d_aterms, aterms.data());
                        htodstream.synchronize();
                    }

                    // Create FFT plan
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
                    }

                    // Events
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;

                    // Power measurement
                    if (local_id == 0) {
                        startStates[device_id] = device.measure();
                    }

                    #pragma omp barrier
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        unsigned int first_bl, last_bl, current_nr_baselines;
                        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                        if (current_nr_baselines == 0) continue;

                        // Initialize iteration
                        auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        void *metadata_ptr        = (void *) plan.get_metadata_ptr(first_bl);
                        void *uvw_ptr             = uvw.data(first_bl, 0);
                        void *visibilities_ptr    = visibilities.data(first_bl, 0, 0);

                        // Power measurement
                        cpuKernels.set_report(report);

                        // Extract subgrid from grid
                        cpuKernels.run_splitter_wstack(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, h_subgrids, grid.data());

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids,
                                auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                auxiliary::sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch FFT
                            executestream.waitEvent(inputReady);
                            device.launch_fft(d_subgrids, ImageDomainToFourierDomain);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device.launch_degridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
        					dtohstream.record(outputFree);

                            device.enqueue_report(dtohstream, current_nr_timesteps, current_nr_subgrids);
                        }

                        outputFree.synchronize();
                    } // end for bl

                    // Wait for all jobs to finish
                    dtohstream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        endStates[device_id] = device.measure();
                    }
                } // end omp parallel

                // End timing
                endStates[nr_devices] = hostPowerSensor->read();
                report.update_host(startStates[nr_devices], endStates[nr_devices]);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auto total_nr_subgrids          = plan.get_nr_subgrids();
                auto total_nr_timesteps         = plan.get_nr_timesteps();
                auto total_nr_visibilities      = plan.get_nr_visibilities();
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                startStates.pop_back(); endStates.pop_back();
                report.print_devices(startStates, endStates);
                report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
                std::clog << std::endl;
                #endif
            } // end degridding

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#include "GenericOptimizedC.h"
