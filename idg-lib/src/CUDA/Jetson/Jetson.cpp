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
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, append(flags), info)
            {
                #if defined(DEBUG)
                cout << "Jetson::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            Compilerflags Jetson::append(Compilerflags flags) {
                stringstream new_flags;
                new_flags << flags;
                new_flags << " -DMAX_NR_TIMESTEPS_GRIDDER=" << GridderMaxwell::max_nr_timesteps;
                new_flags << " -DMAX_NR_TIMESTEPS_DEGRIDDER=" << DegridderMaxwell::max_nr_timesteps;
                new_flags << " -DNR_THREADS_DEGRIDDER=" << DegridderMaxwell::nr_threads;
                return new_flags.str();
            }

            void Jetson::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
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

                // Copy grid to host
                //stream.memcpyDtoHAsync(h_grid, d_grid, sizeof_grid());
                powerRecords[3].enqueue(stream);
                stream.synchronize();

                #if defined(REPORT_TOTAL)
                auxiliary::report("   fft",
                                  PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                std::cout << std::endl;
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
                cu::Module *module_fft = (modules[which_module[name_fft]]);

                // Initialize metadata
                auto max_nr_timesteps = kernel_gridder->get_max_nr_timesteps();
                auto plan = create_plan(uvw, wavenumbers, baselines,
                                        aterm_offsets, kernel_size,
                                        max_nr_timesteps);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cu::Context &context = get_context();
                cu::Stream executestream;
                const int nr_streams = 3;

                // Host memory
                //cu::HostMemory h_wavenumbers((void *) wavenumbers, sizeof_wavenumbers(), CU_MEMHOSTALLOC_DEVICEMAP);
                //cu::HostMemory h_spheroidal((void *) spheroidal, sizeof_spheroidal(), CU_MEMHOSTALLOC_DEVICEMAP);
                //cu::HostMemory h_aterm((void *) aterm, sizeof_aterm(), CU_MEMHOSTALLOC_DEVICEMAP);
                //cu::HostMemory h_grid((void *) grid, sizeof_grid(), CU_MEMHOSTALLOC_DEVICEMAP);

                // Device memory
                //cu::DeviceMemory d_wavenumbers(h_wavenumbers);
                cu::DeviceMemory d_wavenumbers((void *) wavenumbers);
                //cu::DeviceMemory d_spheroidal(h_spheroidal);
                cu::DeviceMemory d_spheroidal((void *) spheroidal);
                //cu::DeviceMemory d_aterm(h_aterm);
                cu::DeviceMemory d_aterm((void *) aterm);
                //cu::DeviceMemory d_grid(h_grid);
                cu::DeviceMemory d_grid((void *) grid);

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;
                PowerSensor::State startState = powerSensor.read();

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event executeFinished;
                    unique_ptr<GridFFT> kernel_fft = get_kernel_fft();

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);

                        // Private host memory
                        //cu::HostMemory h_visibilities(visibilities_ptr, sizeof_visibilities(current_nr_baselines), CU_MEMHOSTALLOC_DEVICEMAP);
                        //cu::HostMemory h_uvw(uvw_ptr, sizeof_uvw(current_nr_baselines), CU_MEMHOSTALLOC_DEVICEMAP);
                        //cu::HostMemory h_metadata(metadata_ptr, sizeof_metadata(current_nr_subgrids), CU_MEMHOSTALLOC_DEVICEMAP);

                        // Private device memory
                        //cu::DeviceMemory d_uvw(h_uvw);
                        //cu::DeviceMemory d_visibilities(h_visibilities);
                        //cu::DeviceMemory d_metadata(h_metadata);
                        //cu::DeviceMemory d_subgrids(sizeof_subgrids(current_nr_subgrids));
                        cu::DeviceMemory d_uvw(uvw_ptr);
                        cu::DeviceMemory d_visibilities(visibilities_ptr);
                        cu::DeviceMemory d_metadata(metadata_ptr);
                        cu::DeviceMemory d_subgrids(sizeof_subgrids(current_nr_subgrids));

                        // Power measurement
                        PowerRecord powerRecords[5];

                        // Create FFT plan
                        kernel_fft->plan(subgridsize, current_nr_subgrids);

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                        {
                            // Launch gridder kernel
                            powerRecords[0].enqueue(executestream);
                            kernel_gridder->launch(
                                executestream, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[1].enqueue(executestream);

                            // Launch FFT
                            //kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_nr_subgrids, d_subgrids);
                            powerRecords[3].enqueue(executestream);

                            // Launch adder kernel
                            kernel_adder->launch(
                                executestream, current_nr_subgrids,
                                d_metadata, d_subgrids, d_grid);
                            powerRecords[4].enqueue(executestream);
                            executestream.record(executeFinished);
                        }

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
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                PowerSensor::State stopState = powerSensor.read();
                unique_ptr<GridFFT> kernel_fft = get_kernel_fft();
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
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
                cout << "Not implemented" << endl;
            }
 
            unique_ptr<Gridder> Jetson::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new GridderJetson(*(modules[which_module.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Jetson::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new DegridderJetson(*(modules[which_module.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Jetson::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFTJetson(*(modules[which_module.at(name_fft)]), mParams));
            }

            unique_ptr<Scaler> Jetson::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new ScalerJetson(*(modules[which_module.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Jetson::get_kernel_adder() const {
                return unique_ptr<Adder>(new AdderJetson(*(modules[which_module.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Jetson::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new SplitterJetson(*(modules[which_module.at(name_splitter)]), mParams));
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg
