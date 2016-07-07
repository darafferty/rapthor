#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()

#include "HybridCUDA.h"

using namespace std;
using namespace idg::proxy::cuda;

namespace idg {
    namespace proxy {
        namespace hybrid {

            /// Constructors
            HybridCUDA::HybridCUDA(
                Parameters params) :
                cpu(params), cuda(params)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << params;
                #endif

                mParams = params;
                cuProfilerStart();
            }

            /// Destructor
            HybridCUDA::~HybridCUDA() {
                cuProfilerStop();
            }

            /*
                High level routines
                These routines operate on grids
            */
            void HybridCUDA::grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                std::complex<float> *grid,
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
                PowerSensor *cuda_power_sensor = device->get_powersensor();

                // Constants
                auto nr_time = mParams.get_nr_time();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<idg::kernel::cuda::Gridder> kernel_gridder = device->get_kernel_gridder();
                unique_ptr<idg::kernel::cuda::Scaler> kernel_scaler = device->get_kernel_scaler();
                unique_ptr<idg::kernel::cpu::Adder> kernel_adder = cpu.get_kernel_adder();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

				// Load context
				cu::Context &context = device->get_context();

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;
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
                double total_runtime_gridding = 0;
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;

                // Start gridder
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
                    total_runtime_gridding = -omp_get_wtime();

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
                        LikwidPowerSensor::State powerStates[2];

                        #pragma omp critical (GPU)
                        {
                			// Copy input data to device
                			htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, cuda.sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, cuda.sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, cuda.sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                			// Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                			// Launch gridder kernel
                			executestream.waitEvent(inputReady);
                			executestream.waitEvent(outputFree);
                            device->measure(powerRecords[0], executestream);
                            kernel_gridder->launch(
                                executestream, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                				d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            device->measure(powerRecords[1], executestream);

                			// Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            device->measure(powerRecords[2], executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_nr_subgrids, d_subgrids);
                            device->measure(powerRecords[3], executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                			// Copy subgrid to host
                			dtohstream.waitEvent(outputReady);
                			dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids, cuda.sizeof_subgrids(current_nr_subgrids));
                			dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();

                        // Add subgrid to grid
                        powerStates[0] = cpu.read_power();
                        #pragma omp critical (CPU)
                        {
                            kernel_adder->run(current_nr_subgrids, metadata_ptr, h_subgrids, grid);
                        }
                        powerStates[1] = cpu.read_power();

                        double runtime_gridder = cuda_power_sensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = cuda_power_sensor->seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler  = cuda_power_sensor->seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_adder   = LikwidPowerSensor::seconds(powerStates[0], powerStates[1]);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                     kernel_gridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                                     cuda_power_sensor->Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("    fft", runtime_fft,
                                                     kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                     kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                     cuda_power_sensor->Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report(" scaler", runtime_scaler,
                                                     kernel_scaler->flops(current_nr_subgrids),
                                                     kernel_scaler->bytes(current_nr_subgrids),
                                                     cuda_power_sensor->Watt(powerRecords[2].state, powerRecords[3].state));
                        auxiliary::report("  adder", runtime_adder,
                                                     kernel_adder->flops(current_nr_subgrids),
                                                     kernel_adder->bytes(current_nr_subgrids),
                                                     LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
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
                total_runtime_gridding += omp_get_wtime();
                unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = device->get_kernel_fft();
                uint64_t total_flops_gridder  = kernel_gridder->flops(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, total_nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, total_nr_subgrids);
                uint64_t total_flops_scaler   = kernel_scaler->flops(total_nr_subgrids);
                uint64_t total_bytes_scaler   = kernel_scaler->bytes(total_nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder->flops(total_nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder->bytes(total_nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft;
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|scaler", total_runtime_scaler, total_flops_scaler, total_bytes_scaler);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, 0);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }

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
                PowerSensor *cuda_power_sensor = device->get_powersensor();

                // Constants
                auto nr_time = mParams.get_nr_time();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_degridder();

                // Load kernels
                unique_ptr<idg::kernel::cuda::Degridder> kernel_degridder = device->get_kernel_degridder();
                unique_ptr<idg::kernel::cpu::Splitter> kernel_splitter = cpu.get_kernel_splitter();

                // Initialize metadata
                auto plan = create_plan(uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                auto total_nr_subgrids   = plan.get_nr_subgrids();
                auto total_nr_timesteps  = plan.get_nr_timesteps();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Load context
				cu::Context &context = device->get_context();

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;
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
                        LikwidPowerSensor::State powerStates[2];

                        // Extract subgrid from grid
                        powerStates[0] = cpu.read_power();
                        kernel_splitter->run(current_nr_subgrids, metadata_ptr, h_subgrids, (void *) grid);
                        powerStates[1] = cpu.read_power();

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
                                executestream, current_nr_subgrids, w_offset, nr_channels, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            device->measure(powerRecords[3], executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                            // Copy visibilities to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, cuda.sizeof_visibilities(current_nr_baselines));
                            dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();

                        double runtime_fft       = cuda_power_sensor->seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = cuda_power_sensor->seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_splitter  = LikwidPowerSensor::seconds(powerStates[0], powerStates[1]);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter->flops(current_nr_subgrids),
                                                       kernel_splitter->bytes(current_nr_subgrids),
                                                       LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("      fft", runtime_fft,
                                                       kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                       kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                       cuda_power_sensor->Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder->flops(current_nr_timesteps, current_nr_subgrids),
                                                       kernel_degridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                                       cuda_power_sensor->Watt(powerRecords[2].state, powerRecords[3].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_degridder += runtime_degridder;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_splitter  += runtime_splitter;
                        #endif
                    } // end for s
                }

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
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, 0);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }

            void HybridCUDA::transform(DomainAtoDomainB direction,
                std::complex<float>* grid) {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                cpu.transform(direction, grid);
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg


// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::hybrid::HybridCUDA Hybrid_CUDA;

    Hybrid_CUDA* Hybrid_CUDA_init(
                unsigned int nr_stations,
                unsigned int nr_channels,
                unsigned int nr_time,
                unsigned int nr_timeslots,
                float        imagesize,
                unsigned int grid_size,
                unsigned int subgrid_size)
    {
        idg::Parameters P;
        P.set_nr_stations(nr_stations);
        P.set_nr_channels(nr_channels);
        P.set_nr_time(nr_time);
        P.set_nr_timeslots(nr_timeslots);
        P.set_imagesize(imagesize);
        P.set_subgrid_size(subgrid_size);
        P.set_grid_size(grid_size);

        return new Hybrid_CUDA(P);
    }

    void Hybrid_CUDA_grid(Hybrid_CUDA* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->grid_visibilities(
                (const std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) metadata,
                (std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void Hybrid_CUDA_degrid(Hybrid_CUDA* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->degrid_visibilities(
                (std::complex<float>*) visibilities,
                    (const float*) uvw,
                    (const float*) wavenumbers,
                    (const int*) metadata,
                    (const std::complex<float>*) grid,
                    w_offset,
                    kernel_size,
                    (const std::complex<float>*) aterm,
                    (const int*) aterm_offsets,
                    (const float*) spheroidal);
     }

    void Hybrid_CUDA_transform(Hybrid_CUDA* p,
                    int direction,
                    void *grid)
    {
       if (direction!=0)
           p->transform(idg::ImageDomainToFourierDomain,
                    (std::complex<float>*) grid);
       else
           p->transform(idg::FourierDomainToImageDomain,
                    (std::complex<float>*) grid);
    }

    void Hybrid_CUDA_destroy(Hybrid_CUDA* p) {
       delete p;
    }

} // end extern "C"
