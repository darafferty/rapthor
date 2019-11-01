#include <algorithm> // max_element

#include "Generic.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;
using namespace powersensor;


namespace idg {
    namespace proxy {
        namespace cuda {

            // The maximum number of CUDA streams in any routine
            const int max_nr_streams = 3;

            // Constructor
            Generic::Generic(
                ProxyInfo info) :
                CUDA(info)
            {
                #if defined(DEBUG)
                std::cout << "Generic::" << __func__ << std::endl;
                #endif

                // Initialize host PowerSensor
                hostPowerSensor = get_power_sensor(sensor_host);
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
                std::cout << __func__ << std::endl;
                std::cout << "Transform direction: " << direction << std::endl;
                #endif

                // Constants
                auto grid_size       = grid.get_x_dim();

                // Load device
                InstanceCUDA &device = get_device(0);

                // Initialize
                cu::Stream& stream = device.get_execute_stream();
                device.set_context();

                // Device memory
                cu::DeviceMemory& d_grid = device.allocate_device_grid(grid_size);

                // Performance measurements
                report.initialize(0, 0, grid_size);
                device.set_report(report);
                PowerRecord powerRecords[4];
                State powerStates[4];
                powerStates[0] = hostPowerSensor->read();
                powerStates[2] = device.measure();

                // Perform fft shift
                device.shift(grid);

                // Copy grid to device
                device.measure(powerRecords[0], stream);
                device.copy_htod(stream, d_grid, grid.data(), grid.bytes());
                device.measure(powerRecords[1], stream);

                // Execute fft
                device.launch_grid_fft(d_grid, grid_size, direction);

                // Copy grid to host
                device.measure(powerRecords[2], stream);
                device.copy_dtoh(stream, grid.data(), d_grid, grid.bytes());
                device.measure(powerRecords[3], stream);
                stream.synchronize();

                // Perform fft shift
                device.shift(grid);

                // Perform fft scaling
                std::complex<float> scale = std::complex<float>(2.0/(grid_size*grid_size), 0);
                if (direction == FourierDomainToImageDomain) {
                    device.scale(grid, scale);
                }

                // End measurements
                stream.synchronize();
                powerStates[1] = hostPowerSensor->read();
                powerStates[3] = device.measure();

                // Report performance
                report.update_input(powerRecords[0].state, powerRecords[1].state);
                report.update_output(powerRecords[2].state, powerRecords[3].state);
                report.update_host(powerStates[0], powerStates[1]);
                report.print_total();
                report.print_device(powerRecords[0].state, powerRecords[3].state);
            } // end transform


            void Generic::do_gridding(
                const Plan& plan,
                const float w_step, // in lambda
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVW<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto nr_correlations = grid.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto image_size      = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 2;

                // Initialize
                initialize(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    aterms, aterms_offsets, spheroidal,
                    max_nr_streams);

                // Page-locked host memory
                InstanceCUDA& device = get_device(0);
                cu::HostMemory& h_visibilities = device.retrieve_host_visibilities();
                cu::HostMemory& h_uvw = device.retrieve_host_uvw();
                Array3D<Visibility<std::complex<float>>> visibilities2(h_visibilities, visibilities.shape());
                Array2D<UVW<float>> uvw2(h_uvw, uvw.shape());
                device.copy_htoh(visibilities2.data(), visibilities.data(), visibilities.bytes());
                device.copy_htoh(uvw2.data(), uvw.data(), uvw.bytes());

                // Allocate grids
                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    if (!m_use_unified_memory) {
                        device.allocate_device_grid(grid.bytes());
                        if (d > 0) {
                            device.allocate_host_grid(grid.bytes());
                        }
                    }
                }

                // Performance measurements
                report.initialize(nr_channels, subgrid_size, grid_size);
                std::vector<State> startStates(nr_devices+1);
                std::vector<State> endStates(nr_devices+1);

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = m_gridding_state.jobsize[device_id];

                    // Initialize device
                    InstanceCUDA& device = get_device(device_id);
                    device.set_context();
                    device.set_report(report);

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers = device.retrieve_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.retrieve_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.retrieve_device_aterms();
                    cu::DeviceMemory& d_aterms_indices       = device.retrieve_device_aterms_indices();
                    cu::DeviceMemory& d_avg_aterm_correction = device.retrieve_device_avg_aterm_correction();
                    cu::DeviceMemory& d_visibilities = device.retrieve_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.retrieve_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.retrieve_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.retrieve_device_metadata(local_id);
                    cu::DeviceMemory& d_grid         = device.retrieve_device_grid();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Copy grid to device / initialize grid to zero
                    if (local_id == 0) {
                        if (!m_use_unified_memory) {
                            if (device_id == 0) {
                                htodstream.memcpyHtoDAsync(d_grid, grid.data(), grid.bytes());
                            } else {
                                d_grid.zero(htodstream);
                            }
                        }
                    }

                    // Events
                    cu::Event inputReady;
                    cu::Event outputReady;

                    // Power measurement
                    if (local_id == 0) {
                        startStates[device_id] = device.measure();
                        startStates[nr_devices] = hostPowerSensor->read();
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
                        void *uvw_ptr             = uvw2.data(first_bl, 0);
                        void *visibilities_ptr    = visibilities2.data(first_bl, 0, 0);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                            auto sizeof_uvw          = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                            auto sizeof_metadata     = auxiliary::sizeof_metadata(current_nr_subgrids);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                            htodstream.record(inputReady);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            device.launch_gridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                                d_aterms, d_aterms_indices, d_avg_aterm_correction, d_metadata, d_subgrids);

                            // Launch FFT
                            device.launch_fft(d_subgrids, FourierDomainToImageDomain);

                            // Launch adder kernel
                            if (m_use_unified_memory) {
                                device.launch_adder_unified(
                                    current_nr_subgrids, grid_size, subgrid_size,
                                    d_metadata, d_subgrids, grid.data());
                            } else {
                                device.launch_adder(
                                    current_nr_subgrids, grid_size, subgrid_size,
                                    d_metadata, d_subgrids, d_grid);
                            }
                            executestream.record(outputReady);
                            device.enqueue_report(executestream, current_nr_timesteps, current_nr_subgrids);
                        }

                        outputReady.synchronize();
                    } // end for bl

                    // Wait for all jobs to finish
                    executestream.synchronize();

                    // End measurement
                    if (local_id == 0) {
                        endStates[device_id] = device.measure();
                    }
                    if (global_id == 0) {
                        endStates[nr_devices] = hostPowerSensor->read();
                        report.update_host(startStates[nr_devices], endStates[nr_devices]);
                    }

                    // Copy grid to host
                    if (!m_use_unified_memory && local_id == 0) {
                        dtohstream.memcpyDtoHAsync(grid.data(), d_grid, auxiliary::sizeof_grid(grid_size));
                    }

                    dtohstream.synchronize();
                    endStates[nr_devices] = hostPowerSensor->read();
                } // end omp parallel

                if (!m_use_unified_memory) {
                    // Add grids
                    for (unsigned d = 1; d < get_num_devices(); d++) {
                        float2 *grid_src = (float2 *) get_device(d).retrieve_host_grid();
                        float2 *grid_dst = (float2 *) grid.data();

                        #pragma omp parallel for
                        for (unsigned i = 0; i < grid_size * grid_size * nr_correlations; i++) {
                            grid_dst[i] += grid_src[i];
                        }
                    }
                }

                // Report performance
                auto total_nr_subgrids        = plan.get_nr_subgrids();
                auto total_nr_timesteps       = plan.get_nr_timesteps();
                auto total_nr_visibilities    = plan.get_nr_visibilities();
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                startStates.pop_back(); endStates.pop_back();
                report.print_devices(startStates, endStates);
                report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
            } // end gridding


            void Generic::do_degridding(
                const Plan& plan,
                const float w_step, // in lambda
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVW<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto image_size      = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 3;

                // Initialize
                initialize(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    aterms, aterms_offsets, spheroidal,
                    max_nr_streams);

                // Page-locked host memory
                InstanceCUDA& device = get_device(0);
                cu::HostMemory& h_visibilities = device.retrieve_host_visibilities();
                cu::HostMemory& h_uvw = device.retrieve_host_uvw();
                Array3D<Visibility<std::complex<float>>> visibilities2(h_visibilities, visibilities.shape());
                Array2D<UVW<float>> uvw2(h_uvw, uvw.shape());
                device.copy_htoh(visibilities2.data(), visibilities.data(), visibilities.bytes());
                device.copy_htoh(uvw2.data(), uvw.data(), uvw.bytes());

                // Allocate grids
                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    if (!m_use_unified_memory) {
                        device.allocate_device_grid(grid.bytes());
                        if (d > 0) {
                            device.allocate_host_grid(grid.bytes());
                        }
                    }
                }

                // Performance measurements
                report.initialize(nr_channels, subgrid_size, grid_size);
                std::vector<State> startStates(nr_devices+1);
                std::vector<State> endStates(nr_devices+1);

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = m_gridding_state.jobsize[device_id];

                    // Initialize device
                    InstanceCUDA& device = get_device(device_id);
                    device.set_context();
                    device.set_report(report);

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers  = device.retrieve_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.retrieve_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.retrieve_device_aterms();
                    cu::DeviceMemory& d_aterms_indices = device.retrieve_device_aterms_indices();
                    cu::DeviceMemory& d_visibilities = device.retrieve_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.retrieve_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.retrieve_device_subgrids(local_id);
                    cu::DeviceMemory& d_metadata     = device.retrieve_device_metadata(local_id);
                    cu::DeviceMemory& d_grid         = device.retrieve_device_grid();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Copy grid to device
                    if (local_id == 0) {
                        if (!m_use_unified_memory) {
                            htodstream.memcpyHtoDAsync(d_grid, grid.data(), grid.bytes());
                        }
                        htodstream.synchronize();
                    }

                    // Events
                    cu::Event inputReady;
                    cu::Event outputReady;
                    cu::Event outputFree;

                    // Power measurement
                    if (local_id == 0) {
                        startStates[device_id] = device.measure();
                    }
                    if (global_id == 0) {
                        startStates[nr_devices] = hostPowerSensor->read();
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
                        void *uvw_ptr             = uvw2.data(first_bl, 0);
                        void *visibilities_ptr    = visibilities2.data(first_bl, 0, 0);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            auto sizeof_uvw          = auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
                            auto sizeof_metadata     = auxiliary::sizeof_metadata(current_nr_subgrids);
                            auto sizeof_visibilities = auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels);
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
                            htodstream.record(inputReady);

                            // Initialize visibilities to zero
                            d_visibilities.zero(htodstream);

                            // Launch splitter kernel
                            executestream.waitEvent(inputReady);
                            if (m_use_unified_memory) {
                                device.launch_splitter_unified(
                                    current_nr_subgrids, grid_size, subgrid_size,
                                    d_metadata, d_subgrids, grid.data());
                            } else {
                                device.launch_splitter(
                                    current_nr_subgrids, grid_size, subgrid_size,
                                    d_metadata, d_subgrids, d_grid);
                            }

                            // Launch FFT
                            device.launch_fft(d_subgrids, ImageDomainToFourierDomain);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device.launch_degridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                                d_aterms, d_aterms_indices, d_metadata, d_subgrids);
                            device.enqueue_report(executestream, current_nr_timesteps, current_nr_subgrids);
                            executestream.record(outputReady);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, sizeof_visibilities);
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();
                    } // end for bl

                    // Wait for all jobs to finish
                    dtohstream.synchronize();

                    // End measurement
                    if (local_id == 0) {
                        endStates[device_id] = device.measure();
                    }
                } // end omp parallel

                // End measurement
                endStates[nr_devices] = hostPowerSensor->read();
                report.update_host(startStates[nr_devices], endStates[nr_devices]);

                // Copy visibilities
                device.copy_htoh(visibilities.data(), visibilities2.data(), visibilities.bytes());

                // Report performance
                auto total_nr_subgrids          = plan.get_nr_subgrids();
                auto total_nr_timesteps         = plan.get_nr_timesteps();
                auto total_nr_visibilities      = plan.get_nr_visibilities();
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                startStates.pop_back(); endStates.pop_back();
                report.print_devices(startStates, endStates);
                report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
            } // end degridding

        } // namespace cuda
    } // namespace proxy
} // namespace idg

#include "GenericC.h"
