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
#include "CPU.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

void dummy() {};

namespace idg {

    namespace proxy {

        /// Constructors
        CPU::CPU(
            Compiler compiler,
            Compilerflags flags,
            Parameters params,
            ProxyInfo info)
          : mParams(params),
            mInfo(info)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif

            parameter_sanity_check(); // throws exception if bad parameters
            compile(compiler, flags);
            load_shared_objects();
            find_kernel_functions();
        }


        CPU::CPU(
            CompilerEnvironment cc,
            Parameters params,
            ProxyInfo info)
           : mParams(params),
             mInfo(info)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // find out which compiler to use
            // call CPU(compiler, flags, params, algparams)
            cerr << "Constructor not implemented yet" << endl;
        }


        CPU::~CPU()
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // unload shared objects by ~Module
            for (unsigned int i=0; i<modules.size(); i++) {
                delete modules[i];
            }

            // Delete .so files
            if( mInfo.delete_shared_objects() ) {
                for (auto libname : mInfo.get_lib_names()) {
                    string lib = mInfo.get_path_to_lib() + "/" + libname;
                    remove(lib.c_str());
                }
            }
        }


        ProxyInfo CPU::default_info()
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // Find library path
            Dl_info dl_info;
            dladdr((void *) dummy, &dl_info);

            // Derive name of library and location
            string libdir = dirname((char *) dl_info.dli_fname);
            string bname = basename((char *) dl_info.dli_fname);
            cout << "Module " << bname << " loaded from: " 
                 << libdir << endl;
            // OLD: string  srcdir = libdir + "/idg/CPU/reference";
            string  srcdir = string(IDG_SOURCE_DIR) 
                + "/src/CPU/reference/kernels";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif
 
            // Create temp directory
            char _tmpdir[] = "/tmp/idg-XXXXXX";
            char *tmpdir = mkdtemp(_tmpdir);
            #if defined(DEBUG)
            cout << "Temporary files will be stored in: " << tmpdir << endl;
            #endif
 
            // Create proxy info
            ProxyInfo p;
            p.set_path_to_src(srcdir);
            p.set_path_to_lib(tmpdir);

            string libgridder = "Gridder.so";
            string libdegridder = "Degridder.so";
            string libfft = "FFT.so";
            string libadder = "Adder.so";
            string libsplitter = "Splitter.so";

            p.add_lib(libgridder);
            p.add_lib(libdegridder);
            p.add_lib(libfft);
            p.add_lib(libadder);
            p.add_lib(libsplitter);

            p.add_src_file_to_lib(libgridder, "KernelGridder.cpp");
            p.add_src_file_to_lib(libdegridder, "KernelDegridder.cpp");
            p.add_src_file_to_lib(libfft, "KernelFFT.cpp");
            p.add_src_file_to_lib(libadder, "KernelAdder.cpp");
            p.add_src_file_to_lib(libsplitter, "KernelSplitter.cpp");

            p.set_delete_shared_objects(true);

            return p;
        }


        string CPU::default_compiler() 
        {
#if defined(USING_GNU_CXX_COMPILER)
            return "g++";
#elif defined(USING_INTEL_CXX_COMPILER)
            return "icpc";
#elif defined(USING_CLANG_CXX_COMPILER)
            return "clang";
#elif defined(USING_MSVC_CXX_COMPILER)
            return "not supported so far";
#else 
            return "g++";
#endif
        }
        

        string CPU::default_compiler_flags() 
        {
#if defined(USING_GNU_CXX_COMPILER)
            return "-Wall -O3 -fopenmp -lfftw3 -lfftw3f -lfftw3f_omp";
#elif defined(USING_INTEL_CXX_COMPILER)
            return "TODO: set flags";
#elif defined(USING_CLANG_CXX_COMPILER)
            return "TODO: set flags";
#elif defined(USING_MSVC_CXX_COMPILER)
            return "TODO: set flags";
#else 
            return "-Wall -O3 -fopenmp -lfftw3 -lfftw3f -lfftw3f_omp";
#endif
        }


        /// High level routines
        void CPU::transform(DomainAtoDomainB direction, void* grid)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            cout << "Transform direction: " << direction << endl;
            #endif

            int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;
            run_fft(grid, sign);
        }


        void CPU::grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_gridder(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities,
                        spheroidal, aterm, metadata, subgrids);
        }


        void CPU::add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_adder(jobsize, nr_subgrids, metadata, subgrids, grid);
        }


        void CPU::split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_splitter(jobsize, nr_subgrids, metadata, subgrids, grid);
        }


        void CPU::degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_degridder(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities,
                      spheroidal, aterm, metadata, subgrids);
        }


        /// Low level routines
        void CPU::run_gridder(int jobsize, GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
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



        void CPU::run_adder(int jobsize, ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
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


        void CPU::run_splitter(int jobsize, SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
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


        void CPU::run_degridder(int jobsize, DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
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


        void CPU::run_fft(void *grid, int sign)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
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


        void CPU::compile(Compiler compiler, Compilerflags flags)
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
            string mparameters =  Parameters::definitions(
              mParams.get_nr_stations(),
              mParams.get_nr_baselines(),
              mParams.get_nr_channels(),
              mParams.get_nr_timesteps(),
              mParams.get_nr_timeslots(),
              mParams.get_imagesize(),
              mParams.get_nr_polarizations(),
              mParams.get_grid_size(),
              mParams.get_subgrid_size());

            printf("TODO: CLEANUP\n");
            string parameters = " " + flags + " " + mparameters + " -I ../src/CPU/reference";

            // for each shared libarary: compile the source files and put into *.so file
            // OMP parallel?!
            for (auto libname : mInfo.get_lib_names()) {
                // create shared object "libname"
                string lib = mInfo.get_path_to_lib() + "/" + libname;

                vector<string> source_files = mInfo.get_source_files(libname);

                string source;
                for (auto src : source_files) {
                  source += mInfo.get_path_to_src() + "/" + src + " ";
                } // source = a.cpp b.cpp c.cpp ...

                cout << lib << " " << source << " " << endl;

                runtime::Source(source.c_str()).compile(compiler.c_str(),
                                                        lib.c_str(),
                                                        parameters.c_str());
            } // for each library
        } // compile

        void CPU::parameter_sanity_check()
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            // TODO: create assertions
            // assert: subgrid_size <= grid_size
            // assert: job_size <= ?
            // [...]
        }


        void CPU::load_shared_objects()
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            for (auto libname : mInfo.get_lib_names()) {
                string lib = mInfo.get_path_to_lib() + "/" + libname;

                #if defined(DEBUG)
                cout << "Loading: " << libname << endl;
                #endif

                modules.push_back(new runtime::Module(lib.c_str()));
            }
        }


        /// maps name -> index in modules that contain that symbol
        void CPU::find_kernel_functions()
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            for (unsigned int i=0; i<modules.size(); i++) {
                if (dlsym(*modules[i], kernel::name_gridder.c_str())) {
                  // found gridder kernel in module i
                  which_module[kernel::name_gridder] = i;
                }
                if (dlsym(*modules[i], kernel::name_degridder.c_str())) {
                  // found degridder kernel in module i
                  which_module[kernel::name_degridder] = i;
                }
                if (dlsym(*modules[i], kernel::name_fft.c_str())) {
                  // found fft kernel in module i
                  which_module[kernel::name_fft] = i;
                }
                if (dlsym(*modules[i], kernel::name_adder.c_str())) {
                  // found adder kernel in module i
                  which_module[kernel::name_adder] = i;
                }
                if (dlsym(*modules[i], kernel::name_splitter.c_str())) {
                  // found gridder kernel in module i
                  which_module[kernel::name_splitter] = i;
                }
            } // end for
        } // end find_kernel_functions

} // namespace proxy

} // namespace idg
