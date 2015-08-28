// TODO: check which include files are really necessary
#include <cstdio> // remove()
#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "idg-config.h"
#include "KNCOffload.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif
#include "Kernels.h"

using namespace std;

namespace idg {

    namespace proxy {

        /// Constructors
        KNCOffload::KNCOffload(Parameters params)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            cout << params;
            #endif

            mParams = params;
        }

        /// High level routines
        void KNCOffload::transform(DomainAtoDomainB direction, void* grid)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            cout << "Transform direction: " << direction << endl;
            #endif

            int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;
            run_fft(grid, sign);
        }


        void KNCOffload::grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_gridder(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities,
                        spheroidal, aterm, metadata, subgrids);
        }


        void KNCOffload::add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_adder(jobsize, nr_subgrids, metadata, subgrids, grid);
        }


        void KNCOffload::split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_splitter(jobsize, nr_subgrids, metadata, subgrids, grid);
        }


        void KNCOffload::degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_degridder(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities,
                      spheroidal, aterm, metadata, subgrids);
        }


        /// Low level routines
        void KNCOffload::run_gridder(int jobsize, GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // Performance measurements
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            double runtime, runtime_gridder, runtime_fft;
            double total_runtime_gridder = 0;
            double total_runtime_fft = 0;
            #endif

            // Constants
            auto nr_stations = mParams.get_nr_stations();
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_channels = mParams.get_nr_channels();
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();
            auto imagesize = mParams.get_imagesize();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            // Number of elements in static
            int wavenumbers_elements = nr_channels;
            int spheroidal_elements  = subgridsize * subgridsize;
            int aterm_elements       = nr_stations * nr_timeslots * nr_polarizations * subgridsize * subgridsize;

            // Pointers to static data
            float *wavenumbers_ptr   = (float *) wavenumbers;
            float *spheroidal_ptr    = (float *) spheroidal;
            complex<float> *aterm_ptr = (complex<float> *) aterm;

            // Start gridder
            #pragma omp target data \
                map(to:wavenumbers_ptr[0:wavenumbers_elements]) \
                map(to:spheroidal_ptr[0:spheroidal_elements]) \
                map(to:aterm_ptr[0:aterm_elements])
            {
                for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // Number of elements in batch
                    int uvw_elements          = nr_timesteps * 3;
                    int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                    int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                    int metadata_elements     = 5;

                    // Pointers to data for current batch
                    float *uvw_ptr                   = (float *) uvw + s * uvw_elements;
                    complex<float> *visibilities_ptr = (complex<float>*) visibilities + s * visibilities_elements;
                    complex<float> *subgrids_ptr     = (complex<float>*) subgrids + s * subgrid_elements;
                    int *metadata_ptr                = (int *) metadata + s * metadata_elements;


                    #pragma omp target                                \
                        map(to:uvw_ptr[0:(current_jobsize * uvw_elements)]) \
                        map(to:visibilities_ptr[0:(current_jobsize * visibilities_elements)]) \
                        map(from:subgrids_ptr[0:(current_jobsize * subgrid_elements)]) \
                        map(to:metadata_ptr[0:(current_jobsize * metadata_elements)])
                    {
                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    runtime_gridder = -omp_get_wtime();
                    #endif


                    kernel_gridder(current_jobsize, w_offset, uvw_ptr, wavenumbers_ptr,
                                   visibilities_ptr, spheroidal_ptr, aterm_ptr, 
                                   metadata_ptr, subgrids_ptr, nr_stations, 
                                   nr_timesteps, nr_timeslots, nr_channels, 
                                   subgridsize, imagesize, nr_polarizations);

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    runtime_gridder += omp_get_wtime();
                    total_runtime_gridder += runtime_gridder;
                    runtime_fft = -omp_get_wtime();
                    #endif

                    kernel_fft(subgridsize, current_jobsize, subgrids_ptr, 
                        FFTW_BACKWARD, nr_polarizations);

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    runtime_fft += omp_get_wtime();
                    total_runtime_fft += runtime_fft;
                    #endif

                    }

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("gridder", runtime_gridder,
                                      kernel_gridder_flops(current_jobsize, nr_timesteps, nr_channels, subgridsize, nr_polarizations),
                                      kernel_gridder_bytes(current_jobsize, nr_timesteps, nr_channels, subgridsize, nr_polarizations));
                    auxiliary::report("fft", runtime_fft,
                                      kernel_fft_flops(subgridsize, nr_subgrids, nr_polarizations),
                                      kernel_fft_bytes(subgridsize, nr_subgrids, nr_polarizations));
                    #endif
                } // end for s
            } // end omp target data

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: gridding" << endl;
            auxiliary::report("gridder", total_runtime_gridder,
                kernel_gridder_flops(nr_subgrids, nr_timesteps, nr_channels, subgridsize, nr_polarizations),
                kernel_gridder_bytes(nr_subgrids, nr_timesteps, nr_channels, subgridsize, nr_polarizations));
            auxiliary::report("fft", total_runtime_fft,
                kernel_fft_flops(subgridsize, nr_subgrids, nr_polarizations),
                kernel_fft_bytes(subgridsize, nr_subgrids, nr_polarizations));
            auxiliary::report_runtime(runtime);
            auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
            #endif

        } // run_gridder



        void KNCOffload::run_degridder(int jobsize, DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // Performance measurements
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            double runtime, runtime_degridder, runtime_fft;
            double total_runtime_degridder = 0;
            double total_runtime_fft = 0;
            #endif

            // Constants
            auto nr_stations = mParams.get_nr_stations();
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_channels = mParams.get_nr_channels();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();
            auto imagesize = mParams.get_imagesize();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            // Start degridder
            for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                // Prevent overflow
                jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                // Number of elements in batch
                int uvw_elements          = nr_timesteps * 3;
                int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                int metadata_elements     = 5;
                int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;

                // Pointers to data for current batch
                void *uvw_ptr          = (float *) uvw + s * uvw_elements;
                void *wavenumbers_ptr  = wavenumbers;
                void *visibilities_ptr = (complex<float>*) visibilities + s * visibilities_elements;
                void *spheroidal_ptr   = spheroidal;
                void *aterm_ptr        = aterm;
                void *metadata_ptr     = (int *) metadata + s * metadata_elements;
                void *subgrids_ptr     = (complex<float>*) subgrids + s * subgrid_elements;

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_fft = -omp_get_wtime();
                #endif

                kernel_fft(subgridsize, jobsize, subgrids_ptr, 
                           FFTW_FORWARD, nr_polarizations);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_fft += omp_get_wtime();
                total_runtime_fft += runtime_fft;
                runtime_degridder = -omp_get_wtime();
                #endif

                kernel_degridder(jobsize, w_offset, uvw_ptr, wavenumbers_ptr, 
                visibilities_ptr, spheroidal_ptr, aterm_ptr, metadata_ptr, 
                subgrids_ptr, nr_stations, nr_timesteps, nr_timeslots,
                nr_channels, subgridsize, imagesize, nr_polarizations);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_degridder += omp_get_wtime();
                total_runtime_degridder += runtime_degridder;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("degridder", runtime_degridder,
                kernel_degridder_flops(jobsize, nr_timesteps, 
                nr_channels, subgridsize, nr_polarizations),
                kernel_degridder_bytes(jobsize, nr_timesteps, 
                nr_channels, subgridsize, nr_polarizations));
                auxiliary::report("fft", runtime_fft,
                kernel_fft_flops(subgridsize, nr_subgrids, nr_polarizations),
                kernel_fft_bytes(subgridsize, nr_subgrids, nr_polarizations));
                #endif
            } // end for s

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: degridding" << endl;
            auxiliary::report("degridder", total_runtime_degridder,
                              kernel_degridder_flops(nr_subgrids, nr_timesteps, 
                                                     nr_channels, subgridsize, 
                                                     nr_polarizations),
                              kernel_degridder_bytes(nr_subgrids, nr_timesteps, 
                                                     nr_channels, subgridsize, 
                                                     nr_polarizations));
            auxiliary::report("fft", total_runtime_fft,
                              kernel_fft_flops(subgridsize, nr_subgrids, 
                                               nr_polarizations),
                              kernel_fft_bytes(subgridsize, nr_subgrids, 
                                               nr_polarizations));
            auxiliary::report_runtime(runtime);
            auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
            #endif
        } // run_degridder



        void KNCOffload::run_adder(int jobsize, ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // Performance measurements
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            double runtime, runtime_adder;
            double total_runtime_adder = 0;
            #endif

            // Constants
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();
            auto gridsize = mParams.get_grid_size();
            
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            // Run adder
            for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                // Prevent overflow
                jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s: jobsize;

                // Number of elements in batch
                int metadata_elements = 5;
                int subgrid_elements  = subgridsize * subgridsize * nr_polarizations;

                // Pointer to data for current jobs
                void *metadata_ptr = (int *) metadata + s * metadata_elements;
                void *subgrid_ptr  = (complex<float>*) subgrids + s * subgrid_elements;
                void *grid_ptr     = grid;

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_adder = -omp_get_wtime();
                #endif

                kernel_adder(jobsize, metadata_ptr, subgrid_ptr, grid_ptr, 
                             gridsize, subgridsize, nr_polarizations);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_adder += omp_get_wtime();
                total_runtime_adder += runtime_adder;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("adder", runtime_adder,
                    kernel_adder_flops(jobsize, subgridsize),
                    kernel_adder_bytes(jobsize, subgridsize, nr_polarizations));
                #endif
            } // end for s

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: adding" << endl;
            auxiliary::report("adder", total_runtime_adder,
                              kernel_adder_flops(nr_subgrids, subgridsize),
                              kernel_adder_bytes(nr_subgrids, subgridsize, nr_polarizations));
            auxiliary::report_runtime(runtime);
            auxiliary::report_subgrids(runtime, nr_subgrids);
            clog << endl;
            #endif

        } // run_adder


        void KNCOffload::run_splitter(int jobsize, SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // Performance measurements
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            double runtime, runtime_splitter;
            double total_runtime_splitter = 0;
            #endif

            // Constants
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();
            auto gridsize = mParams.get_grid_size();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            // Run splitter
            for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                // Prevent overflow
                jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                // Number of elements in batch
                int metadata_elements = 5;
                int subgrid_elements  = subgridsize * subgridsize * nr_polarizations;

                // Pointer to data for current jobs
                void *metadata_ptr = (int *) metadata + s * metadata_elements;
                void *subgrid_ptr  = (complex<float>*) subgrids + s * subgrid_elements;
                void *grid_ptr     = grid;

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_splitter = -omp_get_wtime();
                #endif

                kernel_splitter(jobsize, metadata_ptr, subgrid_ptr, grid_ptr, 
                        gridsize, subgridsize, nr_polarizations);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_splitter += omp_get_wtime();
                total_runtime_splitter += runtime_splitter;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("splitter", runtime_splitter,
                                  kernel_splitter_flops(jobsize, subgridsize),
                                  kernel_splitter_bytes(jobsize, subgridsize, nr_polarizations));
                #endif
            } // end for bl

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: splitting" << endl;
            auxiliary::report("splitter", total_runtime_splitter,
                              kernel_splitter_flops(nr_subgrids, subgridsize),
                              kernel_splitter_bytes(nr_subgrids, subgridsize, nr_polarizations));
            auxiliary::report_runtime(runtime);
            auxiliary::report_subgrids(runtime, nr_subgrids);
            clog << endl;
            #endif
        } // run_splitter


        void KNCOffload::run_fft(void *grid, int sign)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            // Performance measurements
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            double runtime;
            #endif

            // Constants
            auto gridsize = mParams.get_grid_size();
            auto nr_polarizations = mParams.get_nr_polarizations();

            // Start fft
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            kernel_fft(gridsize, 1, grid, sign, nr_polarizations);

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            #endif

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            clog << endl;
            clog << "Total: fft" << endl;
            auxiliary::report("fft", runtime,
                              kernel_fft_flops(gridsize, 1, nr_polarizations),
                              kernel_fft_bytes(gridsize, 1, nr_polarizations));
            auxiliary::report_runtime(runtime);
            clog << endl;
            #endif
        } // run_fft


    } // namespace proxy

} // namespace idg
