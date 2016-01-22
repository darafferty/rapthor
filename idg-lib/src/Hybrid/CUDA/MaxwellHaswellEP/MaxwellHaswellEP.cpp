#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()

#include "idg-config.h"
#include "MaxwellHaswellEP.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace hybrid {

            /// Constructors
            MaxwellHaswellEP::MaxwellHaswellEP(
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
            MaxwellHaswellEP::~MaxwellHaswellEP() {
                cuProfilerStop();
            }

            /*
                High level routines
                These routines operate on grids
            */
            void MaxwellHaswellEP::grid_visibilities(
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

                // Initialize metadata
                auto max_nr_timesteps_gridder = cuda.get_max_nr_timesteps_gridder();
                auto plan = create_plan(uvw, wavenumbers, baselines,
                                        aterm_offsets, kernel_size, max_nr_timesteps_gridder);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Constants
                auto nr_time = mParams.get_nr_time();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<idg::kernel::cuda::Gridder> kernel_gridder = cuda.get_kernel_gridder();
                unique_ptr<idg::kernel::cuda::Scaler> kernel_scaler = cuda.get_kernel_scaler();
                unique_ptr<idg::kernel::cpu::Adder> kernel_adder = cpu.get_kernel_adder();

				// Load context
				cu::Context &context = cuda.get_context();

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

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
                total_runtime_gridding = -omp_get_wtime();

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = cuda.get_kernel_fft();

                    // Private host memory
			        cu::HostMemory h_visibilities(cuda.sizeof_visibilities(jobsize));
                    cu::HostMemory h_uvw(cuda.sizeof_uvw(jobsize));

                    // Private device memory
                	cu::DeviceMemory d_visibilities(cuda.sizeof_visibilities(jobsize));
                	cu::DeviceMemory d_uvw(cuda.sizeof_uvw(jobsize));

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
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr();

                        // Private host memory
    			        cu::HostMemory h_subgrids(cuda.sizeof_subgrids(current_nr_subgrids));
    			        cu::HostMemory h_metadata(cuda.sizeof_metadata(current_nr_subgrids));

                        // Private device memory
                        cu::DeviceMemory d_subgrids(cuda.sizeof_subgrids(current_nr_subgrids));
                        cu::DeviceMemory d_metadata(cuda.sizeof_metadata(current_nr_subgrids));

                        // Power measurement
                        cuda::PowerRecord powerRecords[4];
                        LikwidPowerSensor::State powerStates[2];

                        // Copy memory to host memory
                        h_visibilities.set(visibilities_ptr, cuda.sizeof_visibilities(current_nr_baselines));
                        h_uvw.set(uvw_ptr, cuda.sizeof_uvw(current_nr_baselines));
                        h_metadata.set(metadata_ptr, cuda.sizeof_metadata(current_nr_subgrids));

                        #pragma omp critical (GPU)
                        {
                			// Copy input data to device
                			htodstream.waitEvent(inputFree);
                			htodstream.memcpyHtoDAsync(d_visibilities, h_visibilities, cuda.sizeof_visibilities(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_uvw, h_uvw, cuda.sizeof_uvw(current_nr_baselines));
                            htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, cuda.sizeof_metadata(current_nr_subgrids));
                            htodstream.record(inputReady);

                			htodstream.record(inputReady);

                			// Create FFT plan
                            kernel_fft->plan(subgridsize, current_nr_subgrids);

                			// Launch gridder kernel
                			executestream.waitEvent(inputReady);
                			executestream.waitEvent(outputFree);
                            powerRecords[0].enqueue(executestream);
                            kernel_gridder->launch(
                				executestream, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                				d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[1].enqueue(executestream);

                			// Launch FFT
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);

                            // Launch scaler kernel
                            kernel_scaler->launch(
                                executestream, current_nr_subgrids, d_subgrids);
                            powerRecords[3].enqueue(executestream);
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
                            kernel_adder->run(current_nr_subgrids, h_metadata, h_subgrids, grid);
                        }
                        powerStates[1] = cpu.read_power();

                        double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_scaler  = PowerSensor::seconds(powerRecords[2].state, powerRecords[3].state);
                        double runtime_adder   = LikwidPowerSensor::seconds(powerStates[0], powerStates[1]);
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
                unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = cuda.get_kernel_fft();
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_scaler   = kernel_scaler->flops(nr_subgrids);
                uint64_t total_bytes_scaler   = kernel_scaler->bytes(nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder->bytes(nr_subgrids);
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

            void MaxwellHaswellEP::degrid_visibilities(
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
#if 0
                // initialize metadata
                vector<Metadata> _metadata = init_metadata(uvw, wavenumbers, baselines);
                auto nr_subgrids = _metadata.size();
                const int *metadata = (const int *) _metadata.data();

                // Load kernels
                unique_ptr<idg::kernel::cuda::Degridder> kernel_degridder = cuda.get_kernel_degridder();
                unique_ptr<idg::kernel::cpu::Splitter> kernel_splitter = cpu.get_kernel_splitter();

				// Load context
				cu::Context &context = cuda.get_context();

                // Constants
				auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
				auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

			    auto size_visibilities = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*nr_channels*nr_polarizations;
			    auto size_uvw = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*3;
			    auto size_wavenumbers = 1ULL * nr_channels;
			    auto size_aterm = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize;
			    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
			    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
			    auto size_metadata = 1ULL * nr_subgrids*5;
			    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*subgridsize*subgridsize;

                // Shared device memory
			    cu::DeviceMemory d_wavenumbers(sizeof(float) * size_wavenumbers);
			    cu::DeviceMemory d_aterm(sizeof(complex<float>) * size_aterm);
			    cu::DeviceMemory d_spheroidal(sizeof(float) * size_spheroidal);
                d_wavenumbers.set((void *) wavenumbers);
                d_aterm.set((void *) aterm);
                d_spheroidal.set((void *) spheroidal);

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_splitter = 0;
                total_runtime_degridding = -omp_get_wtime();

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = cuda.get_kernel_fft();

                    // Private host memory
			        cu::HostMemory h_visibilities(jobsize * SIZEOF_VISIBILITIES);
                    cu::HostMemory h_uvw(jobsize * SIZEOF_UVW);
			        cu::HostMemory h_subgrids(jobsize * SIZEOF_SUBGRIDS);
			        cu::HostMemory h_metadata(jobsize * SIZEOF_METADATA);

                    // Private device memory
                	cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                	cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) visibilities + s * visibilities_elements;
                        void *metadata_ptr     = (int *) metadata + s * metadata_elements;

                        // Copy memory to host memory
                        h_visibilities.set(visibilities_ptr, current_jobsize * SIZEOF_VISIBILITIES);
                        h_uvw.set(uvw_ptr, current_jobsize * SIZEOF_UVW);
                        h_metadata.set(metadata_ptr, current_jobsize * SIZEOF_METADATA);

                        // Power measurement
                        cuda::PowerRecord powerRecords[3];
                        LikwidPowerSensor::State powerStates[2];

                        // Extract subgrid from grid
                        powerStates[0] = cpu.read_power();
                        #pragma omp critical (CPU)
                        {
                            kernel_splitter->run(current_jobsize, h_metadata, h_subgrids, (void *) grid);
                        }
                        powerStates[1] = cpu.read_power();

                        #pragma omp critical (GPU)
                		{
                			// Copy input data to device
                			htodstream.waitEvent(inputFree);
                            htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids, current_jobsize * SIZEOF_SUBGRIDS);
                			htodstream.memcpyHtoDAsync(d_visibilities, h_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
                			htodstream.memcpyHtoDAsync(d_uvw, h_uvw, current_jobsize * SIZEOF_UVW);
                			htodstream.memcpyHtoDAsync(d_metadata, h_metadata, current_jobsize * SIZEOF_METADATA);
                			htodstream.record(inputReady);

                			// Create FFT plan
                            kernel_fft->plan(subgridsize, current_jobsize);

                			// Launch FFT
                			executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
                            kernel_fft->launch(executestream, d_subgrids, CUFFT_FORWARD);
                            powerRecords[1].enqueue(executestream);

                			// Launch degridder kernel
                			executestream.waitEvent(outputFree);
                            kernel_degridder->launch(
                				executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
                				d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[2].enqueue(executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                            // Copy visibilities to host
                            dtohstream.waitEvent(outputReady);
                            dtohstream.memcpyDtoHAsync(h_visibilities, d_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
                            dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();
                        memcpy(visibilities_ptr, h_visibilities, SIZEOF_VISIBILITIES * current_jobsize);

                        double runtime_fft       = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_splitter  = LikwidPowerSensor::seconds(powerStates[0], powerStates[1]);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter->flops(current_jobsize),
                                                       kernel_splitter->bytes(current_jobsize),
                                                       LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("      fft", runtime_fft,
                                                       kernel_fft->flops(subgridsize, current_jobsize),
                                                       kernel_fft->bytes(subgridsize, current_jobsize),
                                                       PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder->flops(current_jobsize),
                                                       kernel_degridder->bytes(current_jobsize),
                                                       PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_degridder += runtime_degridder;
                        total_runtime_fft       += runtime_fft;
                        total_runtime_splitter  += runtime_splitter;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_degridding += omp_get_wtime();
                unique_ptr<idg::kernel::cuda::GridFFT> kernel_fft = cuda.get_kernel_fft();
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_splitter   = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter->bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, 0);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
#endif
            }

            void MaxwellHaswellEP::transform(DomainAtoDomainB direction,
                std::complex<float>* grid) {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;

                // Load kernel
                unique_ptr<idg::kernel::cpu::GridFFT> kernel_fft = cpu.get_kernel_fft();

                // Constants
				auto gridsize = mParams.get_grid_size();

                // Power measurement
                LikwidPowerSensor::State powerStates[2];

                // Start fft
                powerStates[0] = cpu.read_power();
                kernel_fft->run(gridsize, 1, grid, sign);
                powerStates[1] = cpu.read_power();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auxiliary::report("|grid_fft",
                                  LikwidPowerSensor::seconds(powerStates[0], powerStates[1]),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                clog << endl;
                #endif

            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::hybrid::MaxwellHaswellEP Hybrid_MaxwellHaswellEP;

    Hybrid_MaxwellHaswellEP* Hybrid_MaxwellHaswellEP_init(
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

        return new Hybrid_MaxwellHaswellEP(P);
    }

    void Hybrid_MaxwellHaswellEP_grid(Hybrid_MaxwellHaswellEP* p,
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

    void Hybrid_MaxwellHaswellEP_degrid(Hybrid_MaxwellHaswellEP* p,
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

    void Hybrid_MaxwellHaswellEP_transform(Hybrid_MaxwellHaswellEP* p,
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

    void Hybrid_MaxwellHaswellEP_destroy(Hybrid_MaxwellHaswellEP* p) {
       delete p;
    }

}  // end extern "C"
