#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <fstream>
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
            Compilerflags flags)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
            // Get context
        	cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);
        
        	// Get devices
        	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            device = devices[deviceNumber];

            mParams = params;
            parameter_sanity_check(); // throws exception if bad parameters
            compile(flags);
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

#if 0
        ProxyInfo OpenCL::default_proxyinfo(string srcdir, string tmpdir) {
            ProxyInfo p;
            p.set_path_to_src(srcdir);
            p.set_path_to_lib(tmpdir);

            string libgridder = "Gridder";
            //string libdegridder = "Degridder";

            p.add_lib(libgridder);
            //p.add_lib(libdegridder);

            p.add_src_file_to_lib(libgridder, "KernelGridder.cl");
            //p.add_src_file_to_lib(libdegridder, "KernelDegridder.cl");

            which_program[kernel::name_gridder] = 0;
            which_program[kernel::name_degridder] = 1;

            return p;
        }
#endif

        string OpenCL::default_compiler_flags() {
            return "-cl-fast-relaxed-math";
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
            kernel::Gridder kernel_gridder(*programs[which_program[kernel::name_gridder]], mParams);
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

        void OpenCL::compile(Compilerflags flags)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            // Source directory
            string srcdir = string(IDG_SOURCE_DIR) 
                + "/src/OpenCL/Reference/kernels";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif
 
            // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
            string mparameters = Parameters::definitions(
              mParams.get_nr_stations(),
              mParams.get_nr_baselines(),
              mParams.get_nr_channels(),
              mParams.get_nr_timesteps(),
              mParams.get_nr_timeslots(),
              mParams.get_imagesize(),
              mParams.get_nr_polarizations(),
              mParams.get_grid_size(),
              mParams.get_subgrid_size());

            string parameters = " " + flags + 
                                " " + "-I " + srcdir +
                                " " + mparameters;

            // Create vector of devices
            std::vector<cl::Device> devices;
            devices.push_back(device);

            // Get context
            cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);

            vector<string> v;
            v.push_back("KernelGridder.cl");

            // Build OpenCL programs
            #pragma omp parallel for
            for (int i = 0; i < v.size(); i++) {
                //string libname = v[i];
                // create shared object "libname"
                //string lib = mInfo.get_path_to_lib() + "/" + libname;

                // Get source filename (vector should only have one element)
                //vector<string> source_files = mInfo.get_source_files(libname);
                string source_file_name = srcdir + "/" + v[i];

                // Read source from file
                ifstream source_file(source_file_name.c_str());
                string source(std::istreambuf_iterator<char>(source_file),
                                  (std::istreambuf_iterator<char>()));
                source_file.close();

                cl::Program program(context, source);
                try {
                    program.build(devices, parameters.c_str());
                    programs.push_back(&program);
                    std::string msg;
                    program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
                    cout << msg;
                } catch (cl::Error &error) {
                    if (strcmp(error.what(), "clBuildProgram") == 0) {
                        std::string msg;
                        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
                        std::cerr << msg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
            } // for each library
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

    } // namespace proxy

} // namespace idg
