#include <cuda.h>
#include <cudaProfiler.h>

#include "HybridCUDA.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace hybrid {

            // Constructor
            HybridCUDA::HybridCUDA(
                CPU* cpuProxy,
                CompileConstants constants) :
                cpuProxy(cpuProxy),
                CUDA(constants, default_info())
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                #endif

                // Initialize host PowerSensor
                #if defined(HAVE_LIKWID)
                hostPowerSensor = LikwidPowerSensor::create();
                #else
                hostPowerSensor = RaplPowerSensor::create();
                #endif

                omp_set_nested(true);

                cuProfilerStart();
            }

            // Destructor
            HybridCUDA::~HybridCUDA() {
                delete cpuProxy;
                delete hostPowerSensor;
                cuProfilerStop();
            }

            /* High level routines */
            void HybridCUDA::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                cpuProxy->transform(direction, grid);
            } // end transform


            void HybridCUDA::do_gridding(
                const Plan& plan,
                const float w_offset, // in lambda
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

                InstanceCPU& cpuKernels = cpuProxy->get_kernels();

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
                auto nr_w_layers  = grid.get_nr_w_layers();
                auto image_size   = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 3;

                // Initialize metadata
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, nr_streams);

                // Initialize memory
                for (int d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    device.set_context();
                    cu::Stream&       htodstream    = device.get_htod_stream();

                    cu::DeviceMemory& d_wavenumbers = device.allocate_device_wavenumbers(nr_channels);
                    cu::DeviceMemory& d_spheroidal  = device.allocate_device_spheroidal(subgrid_size);
                    cu::DeviceMemory& d_aterms      = device.allocate_device_aterms(nr_stations, nr_timeslots, subgrid_size);

                    htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                    htodstream.memcpyHtoDAsync(d_aterms, aterms.data());

                    if (d == 0) {
                        device.reuse_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities.data());
                        device.reuse_host_uvw(nr_baselines, nr_timesteps, uvw.data());
                    }
                }

                // Performance measurements
                double total_runtime_gridder  = 0;
                double total_runtime_fft      = 0;
                double total_runtime_scaler   = 0;
                double total_runtime_adder    = 0;
                double total_runtime_gridding = 0;
                PowerSensor::State startStates[nr_devices+1];
                PowerSensor::State stopStates[nr_devices+1];
                startStates[nr_devices] = hostPowerSensor->read();

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
                    InstanceCUDA& device0 = get_device(0);
                    InstanceCUDA& device  = get_device(device_id);
                    device.set_context();

                    // Load memory objects
                    cu::HostMemory&   h_visibilities = device0.get_host_visibilities();
                    cu::HostMemory&   h_uvw          = device0.get_host_uvw();
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Allocate private memory
                    cu::HostMemory   h_subgrids(device.sizeof_subgrids(max_nr_subgrids, subgrid_size));
                    cu::DeviceMemory d_visibilities(device.sizeof_visibilities(jobsize, nr_timesteps, nr_channels));
                    cu::DeviceMemory d_uvw(device.sizeof_uvw(jobsize, nr_timesteps));
                    cu::DeviceMemory d_subgrids(device.sizeof_subgrids(max_nr_subgrids, subgrid_size));
                    cu::DeviceMemory d_metadata(device.sizeof_metadata(max_nr_subgrids));

                    // Create FFT plan
                    if (local_id == 0) {
                        device.plan_fft(subgrid_size, max_nr_subgrids);
                    }

                    // Events
                    cu::Event inputFree;
                    cu::Event inputReady;
                    cu::Event outputFree;
                    cu::Event outputReady;

                    // Power measurement
                    PowerSensor *devicePowerSensor = device.get_powersensor();
                    PowerRecord powerRecords[5];
                    if (local_id == 0) {
                        startStates[device_id] = device.measure();
                    }

                    #pragma omp barrier
                    #pragma omp single
                    total_runtime_gridding = -omp_get_wtime();
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Initialize iteration
                        auto current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);
                        void *uvw_ptr             = h_uvw.get(bl * device.sizeof_uvw(1, nr_timesteps));
                        void *visibilities_ptr    = h_visibilities.get(bl * device.sizeof_visibilities(1, nr_timesteps, nr_channels));
                        void *metadata_ptr        = (void *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerRecord powerRecords[4];
                        PowerSensor::State powerStates[2];

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device memory
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                                device.sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                device.sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                device.sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch gridder kernel
                            executestream.waitEvent(inputReady);
                            executestream.waitEvent(outputFree);
                            device.measure(powerRecords[0], executestream);
                            device.launch_gridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            device.measure(powerRecords[1], executestream);

                            // Launch FFT
                            device.launch_fft(d_subgrids, FourierDomainToImageDomain);
                            device.measure(powerRecords[2], executestream);

                            // Launch scaler kernel
                            device.launch_scaler(
                                current_nr_subgrids, subgrid_size, d_subgrids);
                            device.measure(powerRecords[3], executestream);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

                            // Copy subgrid to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids,
                                device.sizeof_subgrids(current_nr_subgrids, subgrid_size));
                            dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        // Add subgrids to grid
                        #pragma omp critical (CPU)
                        {
                            powerStates[0]  = hostPowerSensor->read();
                            if (nr_w_layers>1) {
                                cpuKernels.run_adder_wstack(current_nr_subgrids, grid_size, subgrid_size, nr_w_layers, metadata_ptr, h_subgrids, grid.data());
                            } else {
                                cpuKernels.run_adder(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, h_subgrids, grid.data());
                            }
                            powerStates[1]  = hostPowerSensor->read();
                        }

                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", device.flops_gridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                                     device.bytes_gridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                                     devicePowerSensor, powerRecords[0].state, powerRecords[1].state);
                        auxiliary::report("sub-fft", device.flops_fft(subgrid_size, current_nr_subgrids),
                                                     device.bytes_fft(subgrid_size, current_nr_subgrids),
                                                     devicePowerSensor, powerRecords[1].state, powerRecords[2].state);
                        auxiliary::report(" scaler", device.flops_scaler(current_nr_subgrids),
                                                     device.bytes_scaler(current_nr_subgrids),
                                                     devicePowerSensor, powerRecords[2].state, powerRecords[3].state);
                        auxiliary::report("  adder", device.flops_adder(current_nr_subgrids),
                                                     device.bytes_adder(current_nr_subgrids),
                                                     hostPowerSensor, powerStates[0], powerStates[1]);
                        #endif
                        #if defined(REPORT_TOTAL)
                        #pragma omp critical
                        {
                            total_runtime_gridder += devicePowerSensor->seconds(powerRecords[0].state, powerRecords[1].state);
                            total_runtime_fft     += devicePowerSensor->seconds(powerRecords[1].state, powerRecords[2].state);
                            total_runtime_scaler  += devicePowerSensor->seconds(powerRecords[2].state, powerRecords[3].state);
                            total_runtime_adder   += hostPowerSensor->seconds(powerStates[0], powerStates[1]);
                        }
                        #endif
                    } // end for bl

                    // Wait for all jobs to finish
                    executestream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = device.measure();
                    }
                } // end omp parallel

                // End timing
                stopStates[nr_devices]  = hostPowerSensor->read();
                total_runtime_gridding += omp_get_wtime();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                InstanceCUDA& device          = get_device(0);
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
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
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


            void HybridCUDA::do_degridding(
                const Plan& plan,
                const float w_offset, // in lambda
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

                InstanceCPU& cpuKernels = cpuProxy->get_kernels();

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
                auto nr_w_layers  = grid.get_nr_w_layers();
                auto image_size   = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 3;

                // Initialize metadata
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size, nr_streams);

                // Initialize memory
                for (int d = 0; d < get_num_devices(); d++) {
                    InstanceCUDA& device = get_device(d);
                    device.set_context();
                    cu::Stream&       htodstream    = device.get_htod_stream();

                    cu::DeviceMemory& d_wavenumbers = device.allocate_device_wavenumbers(nr_channels);
                    cu::DeviceMemory& d_spheroidal  = device.allocate_device_spheroidal(subgrid_size);
                    cu::DeviceMemory& d_aterms      = device.allocate_device_aterms(nr_stations, nr_timeslots, subgrid_size);

                    htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                    htodstream.memcpyHtoDAsync(d_aterms, aterms.data());

                    if (d == 0) {
                        device.reuse_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities.data());
                        device.reuse_host_uvw(nr_baselines, nr_timesteps, uvw.data());
                    }
                }

                // Performance measurements
                double total_runtime_degridder  = 0;
                double total_runtime_fft        = 0;
                double total_runtime_splitter   = 0;
                double total_runtime_degridding = 0;
                PowerSensor::State startStates[nr_devices+1];
                PowerSensor::State stopStates[nr_devices+1];
                startStates[nr_devices] = hostPowerSensor->read();

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
                    InstanceCUDA& device0 = get_device(0);
                    InstanceCUDA& device  = get_device(device_id);
                    device.set_context();

                    // Load memory objects
                    cu::HostMemory&   h_visibilities = device0.get_host_visibilities();
                    cu::HostMemory&   h_uvw          = device0.get_host_uvw();
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Allocate private memory
                    cu::HostMemory   h_subgrids(device.sizeof_subgrids(max_nr_subgrids, subgrid_size));
                    cu::DeviceMemory d_visibilities(device.sizeof_visibilities(jobsize, nr_timesteps, nr_channels));
                    cu::DeviceMemory d_uvw(device.sizeof_uvw(jobsize, nr_timesteps));
                    cu::DeviceMemory d_subgrids(device.sizeof_subgrids(max_nr_subgrids, subgrid_size));
                    cu::DeviceMemory d_metadata(device.sizeof_metadata(max_nr_subgrids));

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
                    PowerSensor *devicePowerSensor = device.get_powersensor();
                    PowerRecord powerRecords[5];
                    if (local_id == 0) {
                        startStates[device_id] = device.measure();
                    }

                    #pragma omp barrier
                    #pragma omp single
                    total_runtime_degridding = -omp_get_wtime();
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Initialize iteration
                        auto current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);
                        void *uvw_ptr             = h_uvw.get(bl * device.sizeof_uvw(1, nr_timesteps));
                        void *visibilities_ptr    = h_visibilities.get(bl * device.sizeof_visibilities(1, nr_timesteps, nr_channels));
                        void *metadata_ptr        = (void *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerRecord powerRecords[5];
                        PowerSensor::State powerStates[2];

                        // Extract subgrid from grid
                        powerStates[0] = hostPowerSensor->read();
                        if (nr_w_layers>1) {
                            cpuKernels.run_splitter_wstack(current_nr_subgrids, grid_size, subgrid_size, nr_w_layers, metadata_ptr, h_subgrids, grid.data());
                        } else {
                            cpuKernels.run_splitter(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, h_subgrids, grid.data());
                        }
                        powerStates[1] = hostPowerSensor->read();

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids,
                                device.sizeof_subgrids(current_nr_subgrids, subgrid_size));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                device.sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                device.sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                            // Launch FFT
                            device.measure(powerRecords[0], executestream);
                            executestream.waitEvent(inputReady);
                            device.launch_fft(d_subgrids, ImageDomainToFourierDomain);
                            device.measure(powerRecords[1], executestream);

                            // Launch degridder kernel
                            executestream.waitEvent(outputFree);
                            device.measure(powerRecords[2], executestream);
                            device.launch_degridder(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            device.measure(powerRecords[3], executestream);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, device.sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        double runtime_splitter  = hostPowerSensor->seconds(powerStates[0], powerStates[1]);
                        double runtime_fft       = devicePowerSensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = devicePowerSensor->seconds(powerRecords[2].state, powerRecords[3].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", device.flops_adder(current_nr_subgrids),
                                                       device.bytes_adder(current_nr_subgrids),
                                                       hostPowerSensor, powerStates[0], powerStates[1]);
                        auxiliary::report("  sub-fft", device.flops_fft(subgrid_size, current_nr_subgrids),
                                                       device.bytes_fft(subgrid_size, current_nr_subgrids),
                                                       devicePowerSensor, powerRecords[0].state, powerRecords[1].state);
                        auxiliary::report("degridder", device.flops_degridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                                       device.bytes_degridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                                       devicePowerSensor, powerRecords[2].state, powerRecords[3].state);
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter  += hostPowerSensor->seconds(powerStates[0], powerStates[1]);
                        total_runtime_fft       += devicePowerSensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        total_runtime_degridder += devicePowerSensor->seconds(powerRecords[2].state, powerRecords[3].state);
                        #endif
                    } // end for bl

                    // Wait for all jobs to finish
                    dtohstream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = device.measure();
                    }
                } // end omp parallel

                // End timing
                stopStates[nr_devices]    = hostPowerSensor->read();
                total_runtime_degridding += omp_get_wtime();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                InstanceCUDA& device            = get_device(0);
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
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
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


#if 0
            void HybridCUDA::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal) {

                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Get CUDA device
                vector<DeviceInstance*> devices = cuda.get_devices();
                DeviceInstance *device = devices[0];
                PowerSensor *gpu_power_sensor = device->get_powersensor();

                // Configuration
                const int nr_devices = devices.size();
                const int nr_streams = 3;

                // Get CPU power sensor
                PowerSensor *cpu_power_sensor = cpu.get_powersensor();

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();
                auto imagesize = mParams.get_imagesize();

                // Load kernels
                unique_ptr<idg::kernel::cuda::Degridder> kernel_degridder = device->get_kernel_degridder();
                unique_ptr<idg::kernel::cpu::Splitter> kernel_splitter = cpu.get_kernel_splitter();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();
                std::vector<int> jobsize_ = cuda.compute_jobsize(plan, nr_streams);
                int jobsize = jobsize_[0];

                // Initialize
                cu::Context &context      = device->get_context();
                cu::Stream &executestream = device->get_execute_stream();
                cu::Stream &htodstream    = device->get_htod_stream();
                cu::Stream &dtohstream    = device->get_dtoh_stream();
                omp_set_nested(true);

                // Shared host memory
                cu::HostMemory h_visibilities(cuda.sizeof_visibilities(nr_baselines));
                cu::HostMemory h_uvw(cuda.sizeof_uvw(nr_baselines));

                // Copy input data to host memory
                h_visibilities.set((void *) visibilities);
                h_uvw.set((void *) uvw);

                // Shared device memory
                cu::DeviceMemory d_wavenumbers(cuda.sizeof_wavenumbers());
                cu::DeviceMemory d_spheroidal(cuda.sizeof_spheroidal());
                cu::DeviceMemory d_aterm(cuda.sizeof_aterm());

                // Copy static device memory
                htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers);
                htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal);
                htodstream.memcpyHtoDAsync(d_aterm, aterm);
                htodstream.synchronize();

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_splitter = 0;

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = device->get_kernel_fft();

                    // Private host memory
                    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cu::HostMemory h_subgrids(cuda.sizeof_subgrids(max_nr_subgrids));

                    // Private device memory
                    cu::DeviceMemory d_visibilities(cuda.sizeof_visibilities(jobsize));
                    cu::DeviceMemory d_uvw(cuda.sizeof_uvw(jobsize));
                    cu::DeviceMemory d_subgrids(cuda.sizeof_subgrids(max_nr_subgrids));
                    cu::DeviceMemory d_metadata(cuda.sizeof_metadata(max_nr_subgrids));

                    #pragma omp single
                    total_runtime_degridding = -omp_get_wtime();

                    for (int i = 0; i < nr_repetitions; i++) {
                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                        auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + bl * visibilities_elements;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerRecord powerRecords[4];
                        PowerSensor::State powerStates[2];

                        // Extract subgrid from grid
                        powerStates[0] = cpu_power_sensor->read();
                        kernel_splitter->run(current_nr_subgrids, gridsize, metadata_ptr, h_subgrids, (void *) grid);
                        powerStates[1] = cpu_power_sensor->read();

                        #pragma omp critical (GPU)
                		{
                			// Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids, cuda.sizeof_subgrids(current_nr_subgrids));
                            htodstream.memcpyHtoDAsync(d_visibilities, h_visibilities, cuda.sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, cuda.sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, cuda.sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                			// Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                			// Launch FFT
                			executestream.waitEvent(inputReady);
                            device->measure(powerRecords[0], executestream);
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_FORWARD);
                            device->measure(powerRecords[1], executestream);

                			// Launch degridder kernel
                			executestream.waitEvent(outputFree);
                            device->measure(powerRecords[2], executestream);
                            kernel_degridder->launch(
                                executestream, current_nr_subgrids, gridsize, imagesize, w_offset, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            device->measure(powerRecords[3], executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                            // Copy visibilities to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, cuda.sizeof_visibilities(current_nr_baselines));
                            dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();

                        double runtime_fft       = gpu_power_sensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = gpu_power_sensor->seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_splitter  = cpu_power_sensor->seconds(powerStates[0], powerStates[1]);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter->flops(current_nr_subgrids),
                                                       kernel_splitter->bytes(current_nr_subgrids),
                                                       cpu_power_sensor->Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("  sub-fft", runtime_fft,
                                                       kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                       kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                       gpu_power_sensor->Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                       kernel_degridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                                       gpu_power_sensor->Watt(powerRecords[2].state, powerRecords[3].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_degridder += runtime_degridder;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_splitter  += runtime_splitter;
                        #endif
                    } // end for bl
                } // end for repetitions
                } // end omp parallel

                // End runtime measurement
                total_runtime_degridding += omp_get_wtime();

                // Copy visibilities from host memory
                memcpy(visibilities, h_visibilities, cuda.sizeof_visibilities(nr_baselines));

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = device->get_kernel_fft();
                uint64_t total_flops_degridder  = kernel_degridder->flops(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, total_nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, total_nr_subgrids);
                uint64_t total_flops_splitter   = kernel_splitter->flops(total_nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, 0);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding/nr_repetitions, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }
#endif
        } // namespace hybrid
    } // namespace proxy
} // namespace idg
