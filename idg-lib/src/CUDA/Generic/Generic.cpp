#include "Generic.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {

            // Constructor
            Generic::Generic(
                CompileConstants constants,
                ProxyInfo info) :
                CUDA(constants, info)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif

                // Initialize host PowerSensor
                #if defined(HAVE_LIKWID)
                hostPowerSensor = LikwidPowerSensor::create();
                #else
                hostPowerSensor = RaplPowerSensor::create();
                #endif
            }

            // Destructor
            Generic::~Generic() {
                delete hostPowerSensor;
            }

            /* High level routines */
            void Generic::transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto grid_size = grid.get_x_dim();
                auto nr_correlations = mConstants.get_nr_correlations();

                // Load device
                InstanceCUDA &device = get_device(0);
                PowerSensor *devicePowerSensor = device.get_powersensor();

                // Initialize
                cu::Stream& stream = device.get_execute_stream();
                device.set_context();

                // Device memory
                cu::DeviceMemory& d_grid = device.allocate_device_grid(grid_size);

                // Host memory
                cu::HostMemory& h_grid = device.reuse_host_grid(grid_size, grid.data());

                // Performance measurements
                PowerRecord powerRecords[5];
                PowerSensor::State powerStates[4];
                powerStates[0] = hostPowerSensor->read();
                powerStates[2] = devicePowerSensor->read();

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                auto sizeof_grid = device.sizeof_grid(grid_size);
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
                powerStates[3] = devicePowerSensor->read();

                #if defined(REPORT_TOTAL)
                auxiliary::report("     input",
                                  0, sizeof_grid,
                                  devicePowerSensor, powerRecords[0].state, powerRecords[1].state);
                auxiliary::report("  plan-fft",
                                  devicePowerSensor->seconds(powerRecords[1].state, powerRecords[2].state),
                                  0, 0, 0);
                auxiliary::report("  grid-fft",
                                  device.flops_fft(grid_size, 1), device.bytes_fft(grid_size, 1),
                                  devicePowerSensor, powerRecords[2].state, powerRecords[3].state);
                auxiliary::report("    output",
                                  0, sizeof_grid,
                                  devicePowerSensor, powerRecords[3].state, powerRecords[4].state);
                auxiliary::report("  fftshift", time_shift/2, 0, sizeof_grid * 2, 0);
                if (direction == FourierDomainToImageDomain) {
                auxiliary::report("grid-scale", time_scale/2, 0, sizeof_grid * 2, 0);
                }
                auxiliary::report("|host", 0, 0, hostPowerSensor, powerStates[0], powerStates[1]);
                auxiliary::report("|device", 0, 0, devicePowerSensor, powerStates[2], powerStates[3]);
                std::cout << std::endl;
                #endif
            } // end transform


            void Generic::gridding(
                const Plan& plan,
                const float w_step, // in lambda
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Array3D<std::complex<float>>& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

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
                auto image_size   = cell_size * grid_size;

                // Configuration
                const int nr_devices = get_num_devices();
                const int nr_streams = 2;

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
                    cu::DeviceMemory& d_grid        = device.allocate_device_grid(grid_size);

                    htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                    htodstream.memcpyHtoDAsync(d_aterms, aterms.data());

                    if (d == 0) {
                        device.reuse_host_visibilities(nr_baselines, nr_timesteps, nr_channels, visibilities.data());
                        device.reuse_host_uvw(nr_baselines, nr_timesteps, uvw.data());
                        cu::HostMemory& h_grid = device.reuse_host_grid(grid_size, grid.data());
                        htodstream.memcpyHtoDAsync(d_grid, h_grid);
                    } else {
                        cu::HostMemory& h_grid = device.allocate_host_grid(grid_size);
                        d_grid.zero();
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
                    cu::HostMemory&   h_grid         = device.get_host_grid();
                    cu::HostMemory&   h_visibilities = device0.get_host_visibilities();
                    cu::HostMemory&   h_uvw          = device0.get_host_uvw();
                    cu::DeviceMemory& d_wavenumbers  = device.get_device_wavenumbers();
                    cu::DeviceMemory& d_spheroidal   = device.get_device_spheroidal();
                    cu::DeviceMemory& d_aterms       = device.get_device_aterms();
                    cu::DeviceMemory& d_grid         = device.get_device_grid();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Allocate private memory
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
                        PowerRecord powerRecords[5];

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
                            executestream.record(outputReady);
                            executestream.record(inputFree);

                            // Launch adder kernel
                            device.launch_adder(
                                current_nr_subgrids, grid_size, subgrid_size,
                                d_metadata, d_subgrids, d_grid);

                            device.measure(powerRecords[4], executestream);
                            executestream.record(outputReady);
                        }

                        outputReady.synchronize();

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
                                                     devicePowerSensor, powerRecords[3].state, powerRecords[4].state);
                        #endif
                        #if defined(REPORT_TOTAL)
                        #pragma omp critical
                        {
                            total_runtime_gridder += devicePowerSensor->seconds(powerRecords[0].state, powerRecords[1].state);
                            total_runtime_fft     += devicePowerSensor->seconds(powerRecords[1].state, powerRecords[2].state);
                            total_runtime_scaler  += devicePowerSensor->seconds(powerRecords[2].state, powerRecords[3].state);
                            total_runtime_adder   += devicePowerSensor->seconds(powerRecords[3].state, powerRecords[4].state);
                        }
                        #endif
                    } // end for bl

                    // Wait for all jobs to finish
                    executestream.synchronize();

                    // End power measurement
                    if (local_id == 0) {
                        stopStates[device_id] = device.measure();
                    }

                    // Copy grid to host
                    if (local_id == 0) {
                        dtohstream.memcpyDtoHAsync(h_grid, d_grid, device.sizeof_grid(grid_size));
                    }
                    dtohstream.synchronize();
                } // end omp parallel

                // End timing
                stopStates[nr_devices]  = hostPowerSensor->read();
                total_runtime_gridding += omp_get_wtime();

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
            } //end gridding


            void Generic::degridding(
                const Plan& plan,
                const float w_step, // in lambda
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array3D<std::complex<float>>& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

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
                    cu::DeviceMemory& d_grid        = device.allocate_device_grid(grid_size);

                    htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data());
                    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data());
                    htodstream.memcpyHtoDAsync(d_aterms, aterms.data());
                    htodstream.memcpyHtoDAsync(d_grid, grid.data());

                    if (d == 0) {
                        device.reuse_host_grid(grid_size, grid.data());
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
                    cu::DeviceMemory& d_grid         = device.get_device_grid();

                    // Load streams
                    cu::Stream& executestream = device.get_execute_stream();
                    cu::Stream& htodstream    = device.get_htod_stream();
                    cu::Stream& dtohstream    = device.get_dtoh_stream();

                    // Allocate private memory
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

                        #pragma omp critical (lock)
                        {
                            // Copy input data to device
                            htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr,
                                device.sizeof_uvw(current_nr_baselines, nr_timesteps));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr,
                                device.sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

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
                                current_nr_subgrids, grid_size, image_size, w_step, nr_channels, nr_stations,
                                d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
                            device.measure(powerRecords[4], executestream);
                            executestream.record(outputReady);
                            executestream.record(inputFree);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, device.sizeof_visibilities(current_nr_baselines, nr_timesteps, nr_channels));
        					dtohstream.record(outputFree);
                        }

                        outputFree.synchronize();

                        double runtime_splitter  = devicePowerSensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft       = devicePowerSensor->seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_degridder = devicePowerSensor->seconds(powerRecords[3].state, powerRecords[4].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", device.flops_adder(current_nr_subgrids),
                                                       device.bytes_adder(current_nr_subgrids),
                                                       devicePowerSensor, powerRecords[0].state, powerRecords[1].state);
                        auxiliary::report("  sub-fft", device.flops_fft(subgrid_size, current_nr_subgrids),
                                                       device.bytes_fft(subgrid_size, current_nr_subgrids),
                                                       devicePowerSensor, powerRecords[1].state, powerRecords[2].state);
                        auxiliary::report("degridder", device.flops_degridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                                       device.bytes_degridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                                       devicePowerSensor, powerRecords[3].state, powerRecords[4].state);
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_splitter  += devicePowerSensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        total_runtime_fft       += devicePowerSensor->seconds(powerRecords[1].state, powerRecords[2].state);
                        total_runtime_degridder += devicePowerSensor->seconds(powerRecords[3].state, powerRecords[4].state);
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

        } // namespace cuda
    } // namespace proxy
} // namespace idg

#include "GenericC.h"
