#include "idg-config.h"
#include "Jetson.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            /// Constructors
            Jetson::Jetson(
                Parameters params,
                ProxyInfo info) :
                CUDA(params, info)
            {
                #if defined(DEBUG)
                cout << "Jetson::" << __func__ << endl;
                cout << params;
                #endif
            }

            void Jetson::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
#if 0
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();
                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

                // Initialize
                cu::Context &context = get_context();

                // Load kernels
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Initialize
                cu::Stream stream;
                context.setCurrent();

                // Performance measurements
                PowerRecord powerRecords[4];

                // Get device pointer for grid
                cu::DeviceMemory d_grid(grid);

                // Execute fft
                kernel_fft->plan(gridsize, 1);
                powerRecords[0].enqueue(stream);
                kernel_fft->launch(stream, d_grid, sign);
                powerRecords[1].enqueue(stream);
                stream.synchronize();

                #if defined(REPORT_TOTAL)
                auxiliary::report("   fft",
                                  PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                std::cout << std::endl;
                #endif
#endif
            }

            void Jetson::grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
#if 0
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Gridder> kernel_gridder = get_kernel_gridder();
                unique_ptr<Scaler> kernel_scaler = get_kernel_scaler();
                unique_ptr<Adder> kernel_adder = get_kernel_adder();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cu::Context &context = get_context();
                cu::Stream stream;

                // Get device memory pointers
                cu::DeviceMemory d_visibilities((void *) visibilities);
                cu::DeviceMemory d_uvw((void *) uvw);
                cu::DeviceMemory d_wavenumbers((void *) wavenumbers);
                cu::DeviceMemory d_spheroidal((void *) spheroidal);
                cu::DeviceMemory d_aterm((void *) aterm);
                cu::DeviceMemory d_grid((void *) grid);
                cu::DeviceMemory d_metadata(sizeof_metadata(nr_subgrids));
                stream.memcpyHtoDAsync(d_metadata, metadata);
                stream.synchronize();

                // Initialize
                context.setCurrent();
                cu::Event executeFinished;

                // Device memory
                int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                cu::DeviceMemory d_subgrids(sizeof_subgrids(max_nr_subgrids));

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;
                PowerSensor::State startState = powerSensor.read();

                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Compute the number of baselines to process in current iteration
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of elements in batch
                    int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                    int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                    // Number of subgrids for all baselines in job
                    auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                    // Pointers to data for current batch
                    void *uvw_ptr          = d_uvw.get(bl * uvw_elements * sizeof(float));
                    void *visibilities_ptr = d_visibilities.get(bl * visibilities_elements * sizeof(float));
                    void *metadata_ptr     = d_metadata.get(plan.get_metadata_ptr(bl) - plan.get_metadata_ptr(0));

                    // Power measurement
                    PowerRecord powerRecords[5];

                    // Create FFT plan
                    kernel_fft->plan(subgridsize, current_nr_subgrids);

                    // Launch gridder kernel
                    powerRecords[0].enqueue(stream);
                    kernel_gridder->launch(
                        stream, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                        d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                    powerRecords[1].enqueue(stream);

                    // Launch FFT
                    kernel_fft->launch(stream, d_subgrids, CUFFT_INVERSE);
                    powerRecords[2].enqueue(stream);

                    // Launch scaler kernel
                    kernel_scaler->launch(
                        stream, current_nr_subgrids, d_subgrids);
                    powerRecords[3].enqueue(stream);

                    // Launch adder kernel
                    kernel_adder->launch(
                        stream, current_nr_subgrids,
                        d_metadata, d_subgrids, d_grid);
                    powerRecords[4].enqueue(stream);
                    stream.record(executeFinished);

                    executeFinished.synchronize();

                    double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                    double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                    double runtime_scaler  = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                    double runtime_adder   = PowerSensor::seconds(powerRecords[3].state, powerRecords[4].state);
                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("gridder", runtime_gridder,
                                                 kernel_gridder->flops(current_nr_baselines, current_nr_subgrids),
                                                 kernel_gridder->bytes(current_nr_baselines, current_nr_subgrids),
                                                 PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                    auxiliary::report("    fft", runtime_fft,
                                                 kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                 kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                 PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                    auxiliary::report(" scaler", runtime_scaler,
                                                 kernel_scaler->flops(current_nr_subgrids),
                                                 kernel_scaler->bytes(current_nr_subgrids),
                                                 PowerSensor::Watt(powerRecords[2].state, powerRecords[3].state));
                    auxiliary::report("  adder", runtime_adder,
                                                 kernel_adder->flops(current_nr_subgrids),
                                                 kernel_adder->bytes(current_nr_subgrids),
                                                 PowerSensor::Watt(powerRecords[3].state, powerRecords[4].state));
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_gridder += runtime_gridder;
                    total_runtime_fft     += runtime_fft;
                    total_runtime_scaler  += runtime_scaler;
                    total_runtime_adder   += runtime_adder;
                    #endif
                } // end for s
                stream.synchronize();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                PowerSensor::State stopState = powerSensor.read();
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_scaler   = kernel_scaler->flops(nr_subgrids);
                uint64_t total_bytes_scaler   = kernel_scaler->bytes(nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder->bytes(nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft + total_flops_scaler + total_flops_adder;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft + total_bytes_scaler + total_bytes_adder;
                double total_runtime_gridding = PowerSensor::seconds(startState, stopState);
                double total_watt_gridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
#endif
            }

            void Jetson::degrid_visibilities(
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
#if 0
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<Splitter> kernel_splitter = get_kernel_splitter();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
                unique_ptr<Degridder> kernel_degridder = get_kernel_degridder();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cu::Context &context = get_context();
                cu::Stream stream;

                // Get device memory pointers
                cu::DeviceMemory d_visibilities((void *) visibilities);
                cu::DeviceMemory d_uvw((void *) uvw);
                cu::DeviceMemory d_wavenumbers((void *) wavenumbers);
                cu::DeviceMemory d_spheroidal((void *) spheroidal);
                cu::DeviceMemory d_aterm((void *) aterm);
                cu::DeviceMemory d_grid((void *) grid);
                cu::DeviceMemory d_metadata(sizeof_metadata(nr_subgrids));
                d_metadata.set((void *) metadata);

                // Initialize
                context.setCurrent();
                cu::Event executeFinished;

                // Device memory
                int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                cu::DeviceMemory d_subgrids(sizeof_subgrids(max_nr_subgrids));

                // Performance measurements
                double total_runtime_splitter = 0;
                double total_runtime_fft = 0;
                double total_runtime_degridder = 0;
                PowerSensor::State startState = powerSensor.read();

                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Compute the number of baselines to process in current iteration
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of elements in batch
                    int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                    int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                    // Number of subgrids for all baselines in job
                    auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                    // Pointers to data for current batch
                    void *uvw_ptr          = d_uvw.get(bl * uvw_elements * sizeof(float));
                    void *visibilities_ptr = d_visibilities.get(bl * visibilities_elements * sizeof(float));
                    void *metadata_ptr     = d_metadata.get(plan.get_metadata_ptr(bl) - plan.get_metadata_ptr(0));

                    // Power measurement
                    PowerRecord powerRecords[4];

                    // Create FFT plan
                    kernel_fft->plan(subgridsize, current_nr_subgrids);

                    // Launch splitter kernel
                    powerRecords[0].enqueue(stream);
                    kernel_splitter->launch(
                        stream, current_nr_subgrids,
                        d_metadata, d_subgrids, d_grid);
                    powerRecords[1].enqueue(stream);

                    // Launch FFT
                    kernel_fft->launch(stream, d_subgrids, CUFFT_FORWARD);
                    powerRecords[2].enqueue(stream);

                    // Launch degridder kernel
                    kernel_degridder->launch(
                        stream, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                        d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                    powerRecords[3].enqueue(stream);
                    stream.record(executeFinished);

                    executeFinished.synchronize();

                    double runtime_splitter  = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                    double runtime_fft       = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                    double runtime_degridder = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                    #if defined(REPORT_VERBOSE)
                    auxiliary::report(" splitter", runtime_splitter,
                                                   kernel_splitter->flops(current_nr_subgrids),
                                                   kernel_splitter->bytes(current_nr_subgrids),
                                                   PowerSensor::Watt(powerRecords[3].state, powerRecords[4].state));
                    auxiliary::report("      fft", runtime_fft,
                                                   kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                   kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                   PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                    auxiliary::report("degridder", runtime_degridder,
                                                   kernel_degridder->flops(current_nr_baselines, current_nr_subgrids),
                                                   kernel_degridder->bytes(current_nr_baselines, current_nr_subgrids),
                                                   PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_splitter  += runtime_splitter;
                    total_runtime_fft       += runtime_fft;
                    total_runtime_degridder += runtime_degridder;
                    #endif
                } // end for s

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                PowerSensor::State stopState = powerSensor.read();
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_splitter   = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                double total_runtime_degridding = PowerSensor::seconds(startState, stopState);
                double total_watt_degridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, 0);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
#endif
            }

        } // namespace cuda
    } // namespace proxy
} // namespace idg
