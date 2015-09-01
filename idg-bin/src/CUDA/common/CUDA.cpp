#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()
#include <unistd.h> // rmdir()

#include "idg-config.h"
#include "CPU.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {

    namespace proxy {

        /// Constructors
        CUDA::CUDA(
            Parameters params,
            Compiler compiler,
            Compilerflags flags,
            ProxyInfo info)
          : mInfo(info)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif

            mParams = params;
            parameter_sanity_check(); // throws exception if bad parameters
            compile(compiler, flags);
            load_shared_objects();
            find_kernel_functions();
        }

        CUDA::~CUDA()
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // unload shared objects by ~Module
            for (unsigned int i = 0; i < modules.size(); i++) {
                delete modules[i];
            }

            // TODO: Delete .ptx files
            /*
            if (mInfo.delete_shared_objects()) {
                for (auto libname : mInfo.get_lib_names()) {
                    string lib = mInfo.get_path_to_lib() + "/" + libname;
                    remove(lib.c_str());
                }
                rmdir(mInfo.get_path_to_lib().c_str());
            }
            */
        }


        string GPU::make_tempdir() {
            char _tmpdir[] = "/tmp/idg-XXXXXX";
            char *tmpdir = mkdtemp(_tmpdir);
            #if defined(DEBUG)
            cout << "Temporary files will be stored in: " << tmpdir << endl;
            #endif
            return tmpdir;
        }

        ProxyInfo GPU::default_proxyinfo(string srcdir, string tmpdir) {
            ProxyInfo p;
            p.set_path_to_src(srcdir);
            p.set_path_to_lib(tmpdir);

            string libgridder = "Gridder.ptx";
            string libdegridder = "Degridder.ptx";
            string libfft = "FFT.ptx";
            string libadder = "Adder.ptx";
            string libsplitter = "Splitter.ptx";

            p.add_lib(libgridder);
            p.add_lib(libdegridder);
            p.add_lib(libfft);
            p.add_lib(libadder);
            p.add_lib(libsplitter);

            p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
            p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
            p.add_src_file_to_lib(libfft, "KernelFFT.cu");
            p.add_src_file_to_lib(libadder, "KernelAdder.cu");
            p.add_src_file_to_lib(libsplitter, "KernelSplitter.cu");

            p.set_delete_shared_objects(true);

            return p;
        }

        ProxyInfo CUDA::default_info()
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            string srcdir = string(IDG_SOURCE_DIR) 
                + "/src/CUDA/reference/kernels";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif
 
            // Create temp directory
            string tmpdir = make_tempdir();
 
            // Create proxy info
            ProxyInfo p = default_proxyinfo(srcdir, tmpdir);

            return p;
        }


        string CUDA::default_compiler() 
        {
            return "nvcc";
        }
        

        string CPU::default_compiler_flags() 
        {
            #if defined(DEBUG)
            return "-use_fast_math -lineinfo -src-in-ptx";
            #else 
            return "-use_fast_mat";
            #endif
            /*
            TODO
            cu::Device device(deviceNumber);
            int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
                                  device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
            options << " -arch=compute_" << capability;
            options << " -code=sm_" << capability;
            */
        }


        /// High level routines
        void CUDA::transform(DomainAtoDomainB direction, void* grid)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            cout << "Transform direction: " << direction << endl;
            #endif

            int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;
            run_fft(grid, sign);
        }


        void CUDA::grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_gridder(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities,
                        spheroidal, aterm, metadata, subgrids);
        }


        void CUDA::add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_adder(jobsize, nr_subgrids, metadata, subgrids, grid);
        }


        void CUDA::split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_splitter(jobsize, nr_subgrids, metadata, subgrids, grid);
        }


        void CUDA::degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_degridder(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities,
                      spheroidal, aterm, metadata, subgrids);
        }


        /// Low level routines
        void CUDA::run_gridder(int jobsize, GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

       } // run_gridder



        void CUDA::run_adder(int jobsize, ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_adder


        void CUDA::run_splitter(int jobsize, SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
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

            string compiler_parameters;
            #if defined(USING_GNU_CXX_COMPILER)
            compiler_parameters = "-DUSING_GNU_CXX_COMPILER";
            #elif defined(USING_INTEL_CXX_COMPILER)
            compiler_parameters = "-DUSING_INTEL_CXX_COMPILER";
            #endif

            string parameters = " " + flags + " " + mparameters +
                                " " + compiler_parameters;

            vector<string> v = mInfo.get_lib_names();
            #pragma omp parallel for
            for (int i = 0; i < v.size(); i++) {
                string libname = v[i];
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
