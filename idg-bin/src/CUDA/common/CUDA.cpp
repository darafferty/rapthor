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
#include "CUDA.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {

    namespace proxy {

        /// Constructors
        CUDA::CUDA(
            Parameters params,
            unsigned deviceNumber,
            Compiler compiler,
            Compilerflags flags,
            ProxyInfo info)
          : device(deviceNumber),
            mInfo(info)
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

            // Delete .ptx files
            if (mInfo.delete_shared_objects()) {
                for (auto libname : mInfo.get_lib_names()) {
                    string lib = mInfo.get_path_to_lib() + "/" + libname;
                    remove(lib.c_str());
                }
                rmdir(mInfo.get_path_to_lib().c_str());
            }
        }

        string CUDA::make_tempdir() {
            char _tmpdir[] = "/tmp/idg-XXXXXX";
            char *tmpdir = mkdtemp(_tmpdir);
            #if defined(DEBUG)
            cout << "Temporary files will be stored in: " << tmpdir << endl;
            #endif
            return tmpdir;
        }

        ProxyInfo CUDA::default_proxyinfo(string srcdir, string tmpdir) {
            ProxyInfo p;
            p.set_path_to_src(srcdir);
            p.set_path_to_lib(tmpdir);

            string libgridder = "Gridder.ptx";
            string libdegridder = "Degridder.ptx";
            string libadder = "Adder.ptx";
            string libsplitter = "Splitter.ptx";

            p.add_lib(libgridder);
            p.add_lib(libdegridder);
            p.add_lib(libadder);
            p.add_lib(libsplitter);

            p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
            p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
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
                + "/src/CUDA/Reference/kernels";

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
        

        string CUDA::default_compiler_flags() 
        {
            #if defined(DEBUG)
            return "-use_fast_math -lineinfo -src-in-ptx";
            #else 
            return "-use_fast_math";
            #endif
        }


        /// High level routines
        void CUDA::transform(DomainAtoDomainB direction, cu::Context &context, cu::HostMemory &h_grid)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            cout << "Transform direction: " << direction << endl;
            #endif

            int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;
            run_fft(CU_FFT_ARGUMENTS);
        }


        void CUDA::grid_onto_subgrids(int jobsize, CU_GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_gridder(jobsize, CU_GRIDDER_ARGUMENTS);
        }


        void CUDA::add_subgrids_to_grid(int jobsize, CU_ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_adder(jobsize, CU_ADDER_ARGUMENTS);
        }


        void CUDA::split_grid_into_subgrids(int jobsize, CU_SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_splitter(jobsize, CU_SPLITTER_ARGUMENTS);
        }


        void CUDA::degrid_from_subgrids(int jobsize, CU_DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: argument checks

            run_degridder(jobsize, CU_DEGRIDDER_ARGUMENTS);
        }


        /// Low level routines
        void CUDA::run_gridder(int jobsize, CU_GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
       } // run_gridder



        void CUDA::run_adder(int jobsize, CU_ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_adder


        void CUDA::run_splitter(int jobsize, CU_SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_splitter


        void CUDA::run_degridder(int jobsize, CU_DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_degridder


        void CUDA::run_fft(CU_FFT_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_fft


        void CUDA::compile(Compiler compiler, Compilerflags flags)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
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

            // Add device capability
            int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
                                  device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
            string compiler_parameters = " -arch=compute_" + to_string(capability) +
                                         " -code=sm_" + to_string(capability);

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

                cu::Source(source.c_str()).compile(lib.c_str(), parameters.c_str());
            } // for each library
        } // compile

        void CUDA::parameter_sanity_check()
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // TODO: create assertions
            // assert: subgrid_size <= grid_size
            // assert: job_size <= ?
            // [...]
        }


        void CUDA::load_shared_objects()
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            for (auto libname : mInfo.get_lib_names()) {
                string lib = mInfo.get_path_to_lib() + "/" + libname;

                #if defined(DEBUG)
                cout << "Loading: " << libname << endl;
                #endif

                modules.push_back(new cu::Module(lib.c_str()));
            }
        }


        /// maps name -> index in modules that contain that symbol
        void CUDA::find_kernel_functions()
        {
            #if defined(DEBUG)
            cout << "CPU::" << __func__ << endl;
            #endif

            CUfunction function;
            for (unsigned int i=0; i<modules.size(); i++) {
                if (cuModuleGetFunction(&function, *modules[i], kernel::name_gridder.c_str()) == CUDA_SUCCESS) {
                    // found gridder kernel in module i
                    which_module[kernel::name_gridder] = i;
                }
                if (cuModuleGetFunction(&function, *modules[i], kernel::name_degridder.c_str()) == CUDA_SUCCESS) {
                    // found degridder kernel in module i
                    which_module[kernel::name_degridder] = i;
                }
                if (cuModuleGetFunction(&function, *modules[i], kernel::name_adder.c_str()) == CUDA_SUCCESS) {
                    // found adder kernel in module i
                    which_module[kernel::name_adder] = i;
                }
                if (cuModuleGetFunction(&function, *modules[i], kernel::name_splitter.c_str()) == CUDA_SUCCESS) {
                    // found splitter kernel in module i
                    which_module[kernel::name_splitter] = i;
                }
            } // end for
        } // end find_kernel_functions

} // namespace proxy

} // namespace idg
