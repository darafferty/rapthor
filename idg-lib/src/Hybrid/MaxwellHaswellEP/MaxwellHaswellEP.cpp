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
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;
using namespace idg::kernel;

namespace idg {
    namespace proxy {
        namespace hybrid {

            /// Constructors
            MaxwellHaswellEP::MaxwellHaswellEP(
                Parameters params) :
                cpu(params), cuda(params)
            {
                #if defined(DEBUG)
                cout << "Maxwell-HaswellEP::" << __func__ << endl;
                cout << params;
                #endif
            }

            /*
                High level routines
                These routines operate on grids
            */
            void MaxwellHaswellEP::grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *metadata,
                std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal) {
                #if defined(DEBUG)
                cout << "MaxwellHaswellEP::" << __func__ << endl;
                #endif

                // Load kernels
                kernel::cuda::Gridder kernel_gridder = cuda.get_kernel_gridder();
                kernel::cuda::GridFFT kernel_fft_small = cuda.get_kernel_fft();
                kernel::cpu::GridFFT kernel_fft_big = cpu.get_kernel_fft();
                kernel::cpu::Adder kernel_adder = cpu.get_kernel_adder();

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
    			auto nr_subgrids = nr_baselines * nr_timeslots;

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
                double total_runtime_gridding = 0;
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
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
                        h_visibilities.set(visibilities_ptr);
                        h_uvw.set(uvw_ptr);
                        h_metadata.set(metadata_ptr);

                        // Power measurement
                        cuda::PowerRecord powerRecords[3];
                        LikwidPowerSensor::State powerStates[2];

                        #pragma omp critical (GPU)
                		{
                			// Copy input data to device
                			htodstream.waitEvent(inputFree);
                			htodstream.memcpyHtoDAsync(d_visibilities, h_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
                			htodstream.memcpyHtoDAsync(d_uvw, h_uvw, current_jobsize * SIZEOF_UVW);
                			htodstream.memcpyHtoDAsync(d_metadata, h_metadata, current_jobsize * SIZEOF_METADATA);
                			htodstream.record(inputReady);

                			// Create FFT plan
                			kernel_fft_small.plan(subgridsize, current_jobsize);

                			// Launch gridder kernel
                			executestream.waitEvent(inputReady);
                			executestream.waitEvent(outputFree);
                            powerRecords[0].enqueue(executestream);
                			kernel_gridder.launchAsync(
                				executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
                				d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[1].enqueue(executestream);

                			// Launch FFT
                			kernel_fft_small.launchAsync(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                			// Copy subgrid to host
                			dtohstream.waitEvent(outputReady);
                			dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids, current_jobsize * SIZEOF_SUBGRIDS);
                			dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();

                        // Add subgrid to grid
                        powerStates[0] = cpu.read_power();
                        #pragma omp critical (CPU)
                        {
                            kernel_adder.run(jobsize, h_metadata, h_subgrids, grid);
                        }
                        powerStates[1] = cpu.read_power();

                        double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_adder   = LikwidPowerSensor::seconds(powerStates[0], powerStates[1]);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder.flops(current_jobsize),
                                                     kernel_gridder.bytes(current_jobsize),
                                                     PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("    fft", runtime_fft,
                                                     kernel_fft_small.flops(subgridsize, current_jobsize),
                                                     kernel_fft_small.bytes(subgridsize, current_jobsize),
                                                     PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        auxiliary::report("  adder", runtime_adder,
                                                     kernel_adder.flops(current_jobsize),
                                                     kernel_adder.bytes(current_jobsize),
                                                     LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_gridder += runtime_gridder;
                        total_runtime_fft     += runtime_fft;
                        total_runtime_adder   += runtime_adder;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_gridding += omp_get_wtime();
                uint64_t total_flops_gridder  = kernel_gridder.flops(nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder.bytes(nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft_small.flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft_small.bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder.flops(nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder.bytes(nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft;
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, 0);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
            }

            void MaxwellHaswellEP::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *metadata,
                const std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal) {
                #if defined(DEBUG)
                cout << "MaxwellHaswellEP::" << __func__ << endl;
                #endif

                // Load kernels
                kernel::cuda::Degridder kernel_degridder = cuda.get_kernel_degridder();
                kernel::cuda::GridFFT kernel_fft_small = cuda.get_kernel_fft();
                kernel::cpu::GridFFT kernel_fft_big = cpu.get_kernel_fft();
                kernel::cpu::Splitter kernel_splitter = cpu.get_kernel_splitter();

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
    			auto nr_subgrids = nr_baselines * nr_timeslots;

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
                        h_visibilities.set(visibilities_ptr);
                        h_uvw.set(uvw_ptr);
                        h_metadata.set(metadata_ptr);

                        // Power measurement
                        cuda::PowerRecord powerRecords[3];
                        LikwidPowerSensor::State powerStates[2];

                        // Extract subgrid from grid
                        powerStates[0] = cpu.read_power();
                        #pragma omp critical (CPU)
                        {
                            kernel_splitter.run(jobsize, h_metadata, h_subgrids, (void *) grid);
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
                			kernel_fft_small.plan(subgridsize, current_jobsize);

                			// Launch FFT
                			executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
                			kernel_fft_small.launchAsync(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[1].enqueue(executestream);

                			// Launch degridder kernel
                			executestream.waitEvent(outputFree);
                			kernel_degridder.launchAsync(
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

                        double runtime_fft       = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        double runtime_splitter  = LikwidPowerSensor::seconds(powerStates[0], powerStates[1]);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report(" splitter", runtime_splitter,
                                                       kernel_splitter.flops(current_jobsize),
                                                       kernel_splitter.bytes(current_jobsize),
                                                       LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("      fft", runtime_fft,
                                                       kernel_fft_small.flops(subgridsize, current_jobsize),
                                                       kernel_fft_small.bytes(subgridsize, current_jobsize),
                                                       PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("degridder", runtime_degridder,
                                                       kernel_degridder.flops(current_jobsize),
                                                       kernel_degridder.bytes(current_jobsize),
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
                uint64_t total_flops_degridder  = kernel_degridder.flops(nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder.bytes(nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft_small.flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft_small.bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_splitter   = kernel_splitter.flops(nr_subgrids);
                uint64_t total_bytes_splitter   = kernel_splitter.bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, 0);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
            }

            void MaxwellHaswellEP::transform(DomainAtoDomainB direction,
                std::complex<float>* grid) {
                #if defined(DEBUG)
                cout << "MaxwellHaswellEP::" << __func__ << endl;
                #endif
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg
