// TODO: check which include files are really necessary
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
#include "KNCOffload.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {

    namespace proxy {

        /// Constructors
        KNCOffload::KNCOffload(
            Compiler compiler,
            Compilerflags flags,
            Parameters params,
            ProxyInfo info)
            : CPU(compiler, flags, params, info)
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
        }

        ProxyInfo KNCOffload::default_info()
        {
            #if defined(DEBUG)
            cout << "KNCOffload::" << __func__ << endl;
            #endif

            string  srcdir = string(IDG_SOURCE_DIR)
                + "/src/Hybrid/KNC/kernels";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif

            // Create temp directory
            string tmpdir = KNCOffload::make_tempdir();

            // Create proxy info
            ProxyInfo p = KNCOffload::default_proxyinfo(srcdir, tmpdir);

            return p;
        }

        string KNCOffload::default_compiler()
        {
            #if defined(USING_GNU_CXX_COMPILER)
            return "g++";
            #else
            return "icpc";
            #endif
        }

        string KNCOffload::default_compiler_flags()
        {
            #if defined(USING_GNU_CXX_COMPILER)
            return "-Wall -O3 -fopenmp -lfftw3f";
            #else
            return "-Wall -O3 -openmp -mkl -lmkl_avx2 -lmkl_vml_avx2";
            #endif
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
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_channels = mParams.get_nr_channels();
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();

            // load kernel functions
            kernel::Gridder kernel_gridder(*(modules[which_module[kernel::name_gridder]]));
            kernel::GridFFT kernel_fft(*(modules[which_module[kernel::name_fft]]));

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            // Start gridder
            for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                // Prevent overflow
                jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                // Number of elements in batch
                int uvw_elements          = nr_timesteps * 3;
                int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                int metadata_elements     = 5;

                // Pointers to data for current batch
                void *uvw_ptr          = (float *) uvw + s * uvw_elements;
                void *wavenumbers_ptr  = wavenumbers;
                void *visibilities_ptr = (complex<float>*) visibilities + s * visibilities_elements;
                void *spheroidal_ptr   = spheroidal;
                void *aterm_ptr        = aterm;
                void *subgrids_ptr     = (complex<float>*) subgrids + s * subgrid_elements;
                void *metadata_ptr     = (int *) metadata + s * metadata_elements;

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_gridder = -omp_get_wtime();
                #endif

                kernel_gridder.run(jobsize, w_offset, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
                                   spheroidal_ptr, aterm_ptr, metadata_ptr, subgrids_ptr);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_gridder += omp_get_wtime();
                total_runtime_gridder += runtime_gridder;
                runtime_fft = -omp_get_wtime();
                #endif

                kernel_fft.run(subgridsize, jobsize, subgrids_ptr, FFTW_BACKWARD);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_fft += omp_get_wtime();
                total_runtime_fft += runtime_fft;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("gridder", runtime_gridder,
                                  kernel_gridder.flops(jobsize),
                                  kernel_gridder.bytes(jobsize));
                auxiliary::report("fft", runtime_fft,
                                  kernel_fft.flops(subgridsize, nr_subgrids),
                                  kernel_fft.bytes(subgridsize, nr_subgrids));
                #endif
            } // end for s

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: gridding" << endl;
            auxiliary::report("gridder", total_runtime_gridder,
                              kernel_gridder.flops(nr_subgrids),
                              kernel_gridder.bytes(nr_subgrids));
            auxiliary::report("fft", total_runtime_fft,
                              kernel_fft.flops(subgridsize, nr_subgrids),
                              kernel_fft.bytes(subgridsize, nr_subgrids));
            auxiliary::report_runtime(runtime);
            auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
            #endif

        } // run_gridder



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

            // Load kernel function
            kernel::Adder kernel_adder(*(modules[which_module[kernel::name_adder]]));

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

                kernel_adder.run(jobsize, metadata_ptr, subgrid_ptr, grid_ptr);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_adder += omp_get_wtime();
                total_runtime_adder += runtime_adder;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("adder", runtime_adder,
                                  kernel_adder.flops(jobsize),
                                  kernel_adder.bytes(jobsize));
                #endif
            } // end for s

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: adding" << endl;
            auxiliary::report("adder", total_runtime_adder,
                              kernel_adder.flops(nr_subgrids),
                              kernel_adder.bytes(nr_subgrids));
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

            // Load kernel function
            kernel::Splitter kernel_splitter(*(modules[which_module[kernel::name_splitter]]));

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

                kernel_splitter.run(jobsize, metadata_ptr, subgrid_ptr, grid_ptr);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_splitter += omp_get_wtime();
                total_runtime_splitter += runtime_splitter;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("splitter", runtime_splitter,
                                  kernel_splitter.flops(jobsize),
                                  kernel_splitter.bytes(jobsize));
                #endif
            } // end for bl

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: splitting" << endl;
            auxiliary::report("splitter", total_runtime_splitter,
                              kernel_splitter.flops(nr_subgrids),
                              kernel_splitter.bytes(nr_subgrids));
            auxiliary::report_runtime(runtime);
            auxiliary::report_subgrids(runtime, nr_subgrids);
            clog << endl;
            #endif
        } // run_splitter


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
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_channels = mParams.get_nr_channels();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();

            // Load kernel functions
            kernel::Degridder kernel_degridder(*(modules[which_module[kernel::name_degridder]]));
            kernel::GridFFT kernel_fft(*(modules[which_module[kernel::name_fft]]));

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

                kernel_fft.run(subgridsize, jobsize, subgrids_ptr, FFTW_FORWARD);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_fft += omp_get_wtime();
                total_runtime_fft += runtime_fft;
                runtime_degridder = -omp_get_wtime();
                #endif

                kernel_degridder.run(jobsize, w_offset, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
                                     spheroidal_ptr, aterm_ptr, metadata_ptr, subgrids_ptr);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                runtime_degridder += omp_get_wtime();
                total_runtime_degridder += runtime_degridder;
                #endif

                #if defined(REPORT_VERBOSE)
                auxiliary::report("degridder", runtime_degridder,
                kernel_degridder.flops(jobsize),
                kernel_degridder.bytes(jobsize));
                auxiliary::report("fft", runtime_fft,
                kernel_fft.flops(subgridsize, nr_subgrids),
                kernel_fft.bytes(subgridsize, nr_subgrids));
                #endif
            } // end for s

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            clog << endl;
            clog << "Total: degridding" << endl;
            auxiliary::report("degridder", total_runtime_degridder,
                              kernel_degridder.flops(nr_subgrids),
                              kernel_degridder.bytes(nr_subgrids));
            auxiliary::report("fft", total_runtime_fft,
                              kernel_fft.flops(subgridsize, nr_subgrids),
                              kernel_fft.bytes(subgridsize, nr_subgrids));
            auxiliary::report_runtime(runtime);
            auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
            #endif
        } // run_degridder


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

            // Load kernel function
            kernel::GridFFT kernel_fft(*(modules[which_module[kernel::name_fft]]));

            // Start fft
            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime = -omp_get_wtime();
            #endif

            kernel_fft.run(gridsize, 1, grid, sign);

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            runtime += omp_get_wtime();
            #endif

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            clog << endl;
            clog << "Total: fft" << endl;
            auxiliary::report("fft", runtime,
                              kernel_fft.flops(gridsize, 1),
                              kernel_fft.bytes(gridsize, 1));
            auxiliary::report_runtime(runtime);
            clog << endl;
            #endif
        } // run_fft


    } // namespace proxy

} // namespace idg
