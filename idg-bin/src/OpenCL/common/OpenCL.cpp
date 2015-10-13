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
            cl::Context &context,
            unsigned deviceNumber,
            Compilerflags flags) :
            context(context)
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
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
        }

        string OpenCL::default_compiler_flags() {
            return "-cl-fast-relaxed-math";
        }

        /// High level routines
        void OpenCL::transform(DomainAtoDomainB direction, cl::Buffer &h_grid)
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

            // Load kernel functions
            kernel::Gridder kernel_gridder(*programs[which_program[kernel::name_gridder]], mParams);

            // Command queue
            //TODO: seperate queues for htod, dtoh and compute
            cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

            // Performance measurements
            double runtime = 0;

            // Constants
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_channels = mParams.get_nr_channels();
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();

            // Set jobsize
            //TODO: set jobsize according to available memory
            const int jobsize = 8192;

            // Start gridder
            runtime -= omp_get_wtime();
            const int nr_streams = 1;
            #pragma omp parallel num_threads(nr_streams)
            {
                // Events
                cl::Event inputFree;
                cl::Event outputFree;
                cl::Event inputReady;
                cl::Event outputReady;

                // Private device memory
                cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_VISIBILITIES);
                cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_UVW);
                cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_WRITE_ONLY, jobsize * SIZEOF_SUBGRIDS);
                cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_METADATA);
	
                #pragma omp for schedule(dynamic)
                for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // Offsets
                    size_t uvw_offset          = s * nr_timesteps * 3 * sizeof(float);
                    size_t visibilities_offset = s * nr_timesteps * nr_channels * nr_polarizations * sizeof(complex<float>);
                    size_t subgrids_offset     = s * subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>);
                    size_t metadata_offset     = s * 5 * sizeof(int);

                    #pragma omp critical (GPU)
                    {
    						// Copy input data to device
                            queue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, current_jobsize * SIZEOF_UVW, NULL, NULL);
                            queue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, current_jobsize * SIZEOF_VISIBILITIES, NULL, NULL);
                            queue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, current_jobsize * SIZEOF_METADATA, NULL, NULL);
    						// Create FFT plan
                            //TODO

    						// Launch gridder kernel
                            kernel_gridder.launchAsync(queue, current_jobsize, w_offset, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);

    						// Launch FFT
                            //TODO

    						// Copy subgrid to host
                            queue.enqueueCopyBuffer(d_subgrids, h_subgrids, 0, subgrids_offset, current_jobsize * SIZEOF_SUBGRIDS, NULL, &outputReady);
                    }

                    // Wait for device to host transfer to finish
                    outputReady.wait();
                }
            }
            runtime += omp_get_wtime();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            clog << "Total: gridding" << endl;
            clog << "Runtime: " << runtime << " s" << endl;
            auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
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

            // Add all kernels to build
            vector<string> v;
            v.push_back("KernelGridder.cl");

            // Build OpenCL programs
            #pragma omp parallel for
            for (int i = 0; i < v.size(); i++) {
                // Get source filename
                string source_file_name = srcdir + "/" + v[i];

                // Read source from file
                ifstream source_file(source_file_name.c_str());
                string source(std::istreambuf_iterator<char>(source_file),
                                  (std::istreambuf_iterator<char>()));
                source_file.close();

                cl::Program *program =  new cl::Program(context, source);
                try {
                    (*program).build(devices, parameters.c_str());
                    programs.push_back(program);
                    std::string msg;
                    (*program).getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
                    cout << msg;
                } catch (cl::Error &error) {
                    if (strcmp(error.what(), "clBuildProgram") == 0) {
                        std::string msg;
                        (*program).getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
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
        }

    } // namespace proxy

} // namespace idg
