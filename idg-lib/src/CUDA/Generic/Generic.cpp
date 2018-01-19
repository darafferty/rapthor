#include <algorithm> // max_element

#include "Generic.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
    namespace proxy {
        namespace cuda {

            // Constructor
            Generic::Generic(
                ProxyInfo info) :
                CUDA(info)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                // Initialize host PowerSensor
                hostPowerSensor = get_power_sensor(sensor_host);
            }

            // Destructor
            Generic::~Generic() {
                delete hostPowerSensor;
            }

            void Generic::initialize_memory(
                const Plan& plan,
                const std::vector<int> jobsize,
                const int nr_streams,
                const int nr_baselines,
                const int nr_timesteps,
                const int nr_channels,
                const int nr_stations,
                const int nr_timeslots,
                const int subgrid_size,
                const int grid_size,
                void *visibilities,
                void *uvw,
                void *grid)
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
                    cu::DeviceMemory& d_grid        = device.get_device_grid(grid_size);

                    // Dynamic memory (per thread)
                    for (int t = 0; t < nr_streams; t++) {
                        device.get_device_visibilities(t, jobsize[d], nr_timesteps, nr_channels);
                        device.get_device_uvw(t, jobsize[d], nr_timesteps);
                        device.get_device_subgrids(t, max_nr_subgrids, subgrid_size);
                        device.get_device_metadata(t, max_nr_subgrids);
                    }

                    // Host memory
                    if (d == 0) {
                        device.get_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities);
                        device.get_host_uvw(nr_baselines, nr_timesteps, uvw);
                        device.get_host_grid(grid_size, grid);
                    }
                }

            } // end initialize_memory

            /* High level routines */
            void Generic::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto nr_correlations = grid.get_z_dim();;
                auto grid_size       = grid.get_x_dim();

                // Load device
                InstanceCUDA &device = get_device(0);

                // Initialize
                cu::Stream& stream = device.get_execute_stream();
                device.set_context();

                // Device memory
                cu::DeviceMemory& d_grid = device.get_device_grid(grid_size);

                // Host memory
                cu::HostMemory& h_grid = device.get_host_grid(grid_size, grid.data());

                // Performance measurements
                Report report(0, 0, grid_size);
                PowerRecord powerRecords[5];
                State powerStates[4];
                powerStates[0] = hostPowerSensor->read();
                powerStates[2] = device.measure();

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                auto sizeof_grid = auxiliary::sizeof_grid(grid_size);
                device.measure(powerRecords[0], stream);
                stream.memcpyHtoDAsync(d_grid, h_grid, sizeof_grid);
                device.measure(powerRecords[1], stream);

                // Execute fft
                device.plan_fft(grid_size, 1);
                device.measure(powerRecords[2], stream);
                device.launch_fft(d_grid, direction);
                device.measure(powerRecords[3], stream);

                // Copy grid to host
                stream.memcpyDtoHAsync(h_grid, d_grid, sizeof_grid);
                device.measure(powerRecords[4], stream);
                stream.synchronize();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Perform fft scaling
                double time_scale = -omp_get_wtime();
                complex<float> scale = complex<float>(2.0/(grid_size*grid_size), 0);
                if (direction == FourierDomainToImageDomain) {
                    device.scale(grid, scale);
                }
                time_scale += omp_get_wtime();

                // End measurements
                stream.synchronize();
                powerStates[1] = hostPowerSensor->read();
                powerStates[3] = device.measure();

                #if defined(REPORT_TOTAL)
                report.update_input(powerRecords[0].state, powerRecords[1].state);
                report.update_grid_fft(powerRecords[2].state, powerRecords[3].state);
                report.update_output(powerRecords[3].state, powerRecords[4].state);
                report.update_fft_shift(time_shift);
                report.update_fft_scale(time_scale);
                report.update_host(powerStates[0], powerStates[1]);
                report.print_total();
                report.print_device(powerRecords[0].state, powerRecords[4].state);
                clog << endl;
                #endif
            } // end transform


            void Generic::do_gridding(
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
                cout << __func__ << endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
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
                const int nr_streams = 2;

                // Initialize metadata
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, nr_streams);

                // Initialize memory
                initialize_memory(
                    plan, jobsize_, nr_streams,
                    nr_baselines, nr_timesteps, nr_channels, nr_stations, nr_timeslots, subgrid_size, grid_size,
                    visibilities.data(), uvw.data(), grid.data());

                // Performance measurements
                Report report(nr_channels, subgrid_size, 0);
                vector<State> startStates(nr_devices+1);
                vector<State> endStates(nr_devices+1);

                // Locks
                int locks[nr_devices];

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
                    cu::HostMemory&   h_grid         = device.get_host_grid();
                    cu::DeviceMemory& d_grid         = device.get_device_grid();

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
                        if (device_id == 0) {
                            htodstream.memcpyHtoDAsync(d_grid, h_grid);
                        } else {
                            d_grid.zero(htodstream);
                        }
                    }

                    // Create FFT plan
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
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
                        void *uvw_ptr             = uvw.data(first_bl, 0);
                        void *visibilities_ptr    = visibilities.data(first_bl, 0, 0);

                        // Power measurement
                        vector<PowerRecord> powerRecords(5);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                                auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                auxiliary::sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            device.measure(powerRecords[0], executestream);
                            device.launch_gridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            device.measure(powerRecords[1], executestream);

                            // Launch FFT
                            device.launch_fft(d_subgrids, FourierDomainToImageDomain);
                            device.measure(powerRecords[2], executestream);

                            // Launch scaler kernel
                            device.launch_scaler(
                                current_nr_subgrids, subgrid_size, d_subgrids);
                            device.measure(powerRecords[3], executestream);

                            // Launch adder kernel
                            device.launch_adder(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, d_grid);

                            device.measure(powerRecords[4], executestream);
                            executestream.record(outputReady);
                        }

                        outputReady.synchronize();

                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        #pragma omp critical
                        {
                            report.update_gridder(powerRecords[0].state, powerRecords[1].state);
                            report.update_subgrid_fft(powerRecords[1].state, powerRecords[2].state);
                            report.update_scaler(powerRecords[2].state, powerRecords[3].state);
                            report.update_adder(powerRecords[3].state, powerRecords[4].state);
                            report.print(current_nr_timesteps, current_nr_subgrids);
                        }
                        #endif
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
                    if (local_id == 0) {
                        dtohstream.memcpyDtoHAsync(h_grid, d_grid, auxiliary::sizeof_grid(grid_size));
                    }

                    dtohstream.synchronize();
                    endStates[nr_devices] = hostPowerSensor->read();
                } // end omp parallel

                // Add grids
                for (int d = 1; d < get_num_devices(); d++) {
                    float2 *grid_src = (float2 *) get_device(d).get_host_grid();
                    float2 *grid_dst = (float2 *) grid.data();

                    #pragma omp parallel for
                    for (int i = 0; i < grid_size * grid_size * nr_correlations; i++) {
                        grid_dst[i] += grid_src[i];
                    }
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auto total_nr_subgrids        = plan.get_nr_subgrids();
                auto total_nr_timesteps       = plan.get_nr_timesteps();
                auto total_nr_visibilities    = plan.get_nr_visibilities();
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                startStates.pop_back(); endStates.pop_back();
                report.print_devices(startStates, endStates);
                report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
                clog << endl;
                #endif
            } // end gridding


            void Generic::do_degridding(
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
                cout << __func__ << endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
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
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, nr_streams);

                // Initialize memory
                initialize_memory(
                    plan, jobsize_, nr_streams,
                    nr_baselines, nr_timesteps, nr_channels, nr_stations, nr_timeslots, subgrid_size, grid_size,
                    visibilities.data(), uvw.data(), grid.data());

                // Performance measurements
                Report report(nr_channels, subgrid_size, 0);
                vector<State> startStates(nr_devices+1);
                vector<State> endStates(nr_devices+1);

                // Locks
                int locks[nr_devices];

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
                    cu::HostMemory&   h_grid         = device.get_host_grid();
                    cu::DeviceMemory& d_grid         = device.get_device_grid();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Copy static data structures
                    if (local_id == 0) {
                        htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                        htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                        htodstream.memcpyHtoDAsync(d_aterms, aterms.data());
                        htodstream.memcpyHtoDAsync(d_grid, h_grid);
                        htodstream.synchronize();
                    }

                    // Create FFT plan
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
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
                        void *uvw_ptr             = uvw.data(first_bl, 0);
                        void *visibilities_ptr    = visibilities.data(first_bl, 0, 0);

                        // Power measurement
                        vector<PowerRecord> powerRecords(5);

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                auxiliary::sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Initialize visibilities to zero
                            d_visibilities.zero(htodstream);

                            // Launch splitter kernel
                            executestream.waitEvent(inputReady);
                            device.measure(powerRecords[0], executestream);
                            device.launch_splitter(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, d_grid);
                            device.measure(powerRecords[1], executestream);

                            // Launch FFT
                            device.launch_fft(d_subgrids, ImageDomainToFourierDomain);
                            device.measure(powerRecords[2], executestream);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device.measure(powerRecords[3], executestream);
                            device.launch_degridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            device.measure(powerRecords[4], executestream);
                            executestream.record(outputReady);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        #pragma omp critical
                        {
                            report.update_splitter(powerRecords[0].state, powerRecords[1].state);
                            report.update_subgrid_fft(powerRecords[1].state, powerRecords[2].state);
                            report.update_degridder(powerRecords[3].state, powerRecords[4].state);
                            report.print(current_nr_timesteps, current_nr_subgrids);
                        }
                        #endif
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

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auto total_nr_subgrids          = plan.get_nr_subgrids();
                auto total_nr_timesteps         = plan.get_nr_timesteps();
                auto total_nr_visibilities      = plan.get_nr_visibilities();
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                startStates.pop_back(); endStates.pop_back();
                report.print_devices(startStates, endStates);
                report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
                clog << endl;
                #endif
            } // end degridding

        } // namespace cuda
    } // namespace proxy
} // namespace idg

#include "GenericC.h"
