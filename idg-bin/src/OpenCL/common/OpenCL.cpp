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
#include "OpenCL.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {

    namespace proxy {

        /// Constructors
        OpenCL::OpenCL(
            Parameters params,
            unsigned deviceNumber,
            Compiler compiler,
            Compilerflags flags,
            ProxyInfo info)
        {
//          : device(deviceNumber),
//            mInfo(info)
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
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

        OpenCL::~OpenCL()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            #if 0
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
            #endif
        }

        string OpenCL::make_tempdir() {
            char _tmpdir[] = "/tmp/idg-XXXXXX";
            char *tmpdir = mkdtemp(_tmpdir);
            #if defined(DEBUG)
            cout << "Temporary files will be stored in: " << tmpdir << endl;
            #endif
            return tmpdir;
        }

        ProxyInfo OpenCL::default_proxyinfo(string srcdir, string tmpdir) {
            ProxyInfo p;
            p.set_path_to_src(srcdir);
            p.set_path_to_lib(tmpdir);

            #if 0
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
            #endif

            p.set_delete_shared_objects(true);

            return p;
        }

        ProxyInfo OpenCL::default_info()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            string srcdir = string(IDG_SOURCE_DIR) 
                + "/src/OpenCL/Reference/kernels";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif
 
            // Create temp directory
            string tmpdir = make_tempdir();
 
            // Create proxy info
            ProxyInfo p = default_proxyinfo(srcdir, tmpdir);

            return p;
        }


        string OpenCL::default_compiler() 
        {
            return "";
        }
        

        string OpenCL::default_compiler_flags() 
        {
            return "";
        }


        /// High level routines
        void OpenCL::transform(DomainAtoDomainB direction, cl::Context &context, cl::Buffer &h_grid)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            cout << "Transform direction: " << direction << endl;
            #endif

            int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;
            run_fft(CL_FFT_ARGUMENTS);
        }


        void OpenCL::grid_onto_subgrids(CL_GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            run_gridder(CL_GRIDDER_ARGUMENTS);
        }


        void OpenCL::add_subgrids_to_grid(CL_ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            run_adder(CL_ADDER_ARGUMENTS);
        }


        void OpenCL::split_grid_into_subgrids(CL_SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            run_splitter(CL_SPLITTER_ARGUMENTS);
        }


        void OpenCL::degrid_from_subgrids(CL_DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            run_degridder(CL_DEGRIDDER_ARGUMENTS);
        }

        /*
            Size of data structures for a single job
        */
        #define SIZEOF_SUBGRIDS 1ULL * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>)
        #define SIZEOF_UVW      1ULL * nr_timesteps * 3 * sizeof(float)
        #define SIZEOF_VISIBILITIES 1ULL * nr_timesteps * nr_channels * nr_polarizations * sizeof(complex<float>)
        #define SIZEOF_METADATA 1ULL * 5 * sizeof(int)


        /// Low level routines
        /*
            Gridder
        */
        void OpenCL::run_gridder(CL_GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif
        } // run_gridder


        void OpenCL::run_adder(CL_ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif
        } // run_adder


        void OpenCL::run_splitter(CL_SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif
        } // run_splitter


        void OpenCL::run_degridder(CL_DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif
        } // run_degridder


        void OpenCL::run_fft(CL_FFT_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_fft


        void OpenCL::compile(Compiler compiler, Compilerflags flags)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            #if 0
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
            #endif

            #if 0
            // Add device capability
            int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
                                  device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
            string compiler_parameters = " -arch=compute_" + to_string(capability) +
                                         " -code=sm_" + to_string(capability);

            string parameters = " " + flags + 
                                " " + compiler_parameters +
                                " " + mparameters;

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
            #endif
        } // compile

        void OpenCL::parameter_sanity_check()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            // TODO: create assertions
            // assert: subgrid_size <= grid_size
            // assert: job_size <= ?
            // [...]
        }


        void OpenCL::load_shared_objects()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            #if 0
            for (auto libname : mInfo.get_lib_names()) {
                string lib = mInfo.get_path_to_lib() + "/" + libname;

                #if defined(DEBUG)
                cout << "Loading: " << libname << endl;
                #endif

                modules.push_back(new cu::Module(lib.c_str()));
            }
            #endif
        }


        /// maps name -> index in modules that contain that symbol
        void OpenCL::find_kernel_functions()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            #if 0
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
            #endif
        } // end find_kernel_functions

    } // namespace proxy

} // namespace idg
