#include <algorithm> // max_element

#include "UnifiedOptimized.h"


#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;
using namespace powersensor;


/*
 * Option to enable/disable reordering of the grid
 * to the host grid format, rather than the tiled
 * format used in the adder and splitter kernels.
 */
#define ENABLE_TILING 1


namespace idg {
    namespace proxy {
        namespace hybrid {

            // The maximum number of CUDA streams in any routine
            const int max_nr_streams = 3;

            // Constructor
            UnifiedOptimized::UnifiedOptimized(
                ProxyInfo info) :
                CUDA(info),
                memory(0)
            {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                hostPowerSensor = get_power_sensor(sensor_host);
                cpuProxy =  new idg::proxy::cpu::Optimized();
                omp_set_nested(true);
            }

            // Destructor
            UnifiedOptimized::~UnifiedOptimized() {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                free_memory();
                delete cpuProxy;
                delete hostPowerSensor;
            }

            void UnifiedOptimized::initialize_memory(
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
                void *uvw)
            {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                for (unsigned d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    device.set_context();
                    int max_jobsize = * max_element(begin(jobsize), end(jobsize));
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, max_jobsize);

                    // Static memory
                    device.get_device_wavenumbers(nr_channels);
                    device.get_device_spheroidal(subgrid_size);
                    device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);


                    unsigned int avg_aterm_correction_subgrid_size = m_avg_aterm_correction.size() ? subgrid_size : 0;
                    device.get_device_avg_aterm_correction(avg_aterm_correction_subgrid_size);

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
                    }
                }

            } // end initialize_memory


            void UnifiedOptimized::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                cpuProxy->transform(direction, grid);
            } // end transform


            void UnifiedOptimized::do_gridding(
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
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

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
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_stations, nr_timeslots, nr_timesteps, nr_channels, subgrid_size, max_nr_streams);

                // Initialize memory
                initialize_memory(
                    plan, jobsize_, nr_streams,
                    nr_baselines, nr_timesteps, nr_channels, nr_stations, nr_timeslots, subgrid_size, grid_size,
                    visibilities.data(), uvw.data());

                // Performance measurements
                report.initialize(nr_channels, subgrid_size, grid_size);
                vector<State> startStates(nr_devices+1);
                vector<State> endStates(nr_devices+1);

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = jobsize_[device_id];
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                    // Initialize device
                    InstanceCUDA& device  = get_device(device_id);
                    device.set_context();

                    // Load memory objects
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                    cu::DeviceMemory& d_avg_aterm_correction = device.get_device_avg_aterm_correction();
                    cu::DeviceMemory& d_visibilities = device.get_device_visibilities(local_id);
                    cu::DeviceMemory& d_uvw          = device.get_device_uvw(local_id);
                    cu::DeviceMemory& d_subgrids     = device.get_device_subgrids(local_id, max_nr_subgrids, subgrid_size);
                    cu::DeviceMemory& d_metadata     = device.get_device_metadata(local_id, max_nr_subgrids);

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
                        device.set_report(report);
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
                            device.launch_gridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_avg_aterm_correction, d_metadata, d_subgrids);

                            // Launch gridder post-processing kernel
                            device.launch_gridder_post(
                                current_nr_subgrids, subgrid_size, nr_stations,
                                d_spheroidal, d_aterms, d_avg_aterm_correction, d_metadata, d_subgrids);

                            // Launch FFT
                            device.launch_fft(d_subgrids, FourierDomainToImageDomain);

                            // Launch adder kernel
                            device.launch_adder_unified(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, grid.data());
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

                    dtohstream.synchronize();
                    endStates[nr_devices] = hostPowerSensor->read();
                } // end omp parallel

                // Undo tiling
                #if ENABLE_TILING
                const int tile_size = get_device(0).get_tile_size_grid();
                Grid grid_copy(1, nr_correlations, grid_size, grid_size);
                memcpy((void *) grid_copy.data(), grid.data(), grid.bytes());
                get_device(0).tile_backward(tile_size, grid_copy, grid);
                #endif

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


            void UnifiedOptimized::do_degridding(
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
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                // Arguments
                auto nr_baselines    = visibilities.get_z_dim();
                auto nr_timesteps    = visibilities.get_y_dim();
                auto nr_channels     = visibilities.get_x_dim();
                auto nr_stations     = aterms.get_z_dim();
                auto nr_timeslots    = aterms.get_w_dim();
                auto nr_correlations = grid.get_z_dim();
                auto grid_size       = grid.get_x_dim();
                auto image_size      = cell_size * grid_size;

                // Apply tiling
                #if ENABLE_TILING
                const int tile_size = get_device(0).get_tile_size_grid();
                Grid grid_tiled(1, nr_correlations, grid_size, grid_size);
                get_device(0).tile_forward(tile_size, grid, grid_tiled);
                memcpy(grid.data(), grid_tiled.data(), grid.bytes());
                #endif

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 3;

                // Initialize metadata
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_stations, nr_timeslots, nr_timesteps, nr_channels, subgrid_size, max_nr_streams);

                // Initialize memory
                initialize_memory(
                    plan, jobsize_, nr_streams,
                    nr_baselines, nr_timesteps, nr_channels, nr_stations, nr_timeslots, subgrid_size, grid_size,
                    visibilities.data(), uvw.data());

                // Performance measurements
                report.initialize(nr_channels, subgrid_size, grid_size);
                vector<State> startStates(nr_devices+1);
                vector<State> endStates(nr_devices+1);

                #pragma omp parallel num_threads(nr_devices * nr_streams)
                {
                    int global_id = omp_get_thread_num();
                    int device_id = global_id / nr_streams;
                    int local_id  = global_id % nr_streams;
                    int jobsize   = jobsize_[device_id];
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
                        device.set_report(report);
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
                            device.launch_splitter_unified(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, grid.data());

                            // Launch FFT
                            device.launch_fft(d_subgrids, ImageDomainToFourierDomain);

                            // Launch degridder pre-processing kernel
                            device.launch_degridder_pre(
                                current_nr_subgrids, subgrid_size, nr_stations,
                                d_spheroidal, d_aterms, d_metadata, d_subgrids);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device.launch_degridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            executestream.record(outputReady);

                            // Copy visibilities to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
                            dtohstream.record(outputFree);
                            device.enqueue_report(executestream, current_nr_timesteps, current_nr_subgrids);
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

                // Undo tiling
                #if ENABLE_TILING
                get_device(0).tile_backward(tile_size, grid, grid_tiled);
                memcpy(grid.data(), grid_tiled.data(), grid.bytes());
                #endif

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


            /*
             * Methods for memory management
             */
            void* UnifiedOptimized::allocate_memory(
                size_t bytes)
            {
                cu::UnifiedMemory* m = new cu::UnifiedMemory(bytes);
                memory.push_back(m);
                return m->ptr();
            }

            void UnifiedOptimized::free_memory(void *ptr)
            {
                for (unsigned i = 0; i < memory.size(); i++) {
                    cu::UnifiedMemory* m = memory[i];
                    if (m->ptr() == ptr) {
                        memory.erase(memory.begin() + i);
                        delete m;
                        break;
                    }
                }
            }

            void UnifiedOptimized::free_memory() {
                for (unsigned i = 0; i < memory.size(); i++) {
                    delete memory[i];
                }
                memory.clear();
            }

            Grid UnifiedOptimized::get_grid(
                size_t nr_w_layers,
                size_t nr_correlations,
                size_t height,
                size_t width)
            {
                assert(height == width);
                size_t bytes = nr_w_layers * auxiliary::sizeof_grid(height, nr_correlations);
                std::complex<float>* ptr = (std::complex<float> *) allocate_memory(bytes);
                return Grid(ptr, nr_w_layers, nr_correlations, height, width);
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#include "UnifiedOptimizedC.h"
