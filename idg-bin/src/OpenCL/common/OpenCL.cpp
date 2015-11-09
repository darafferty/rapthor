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

        static PowerSensor *powerSensor;

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

            #if defined(MEASURE_POWER_ARDUINO)
            cout << "Opening power sensor: " << STR_POWER_SENSOR << endl;
            cout << "Writing power consumption to file: " << STR_POWER_FILE << endl;
            powerSensor = new PowerSensor(STR_POWER_SENSOR, STR_POWER_FILE);
            #endif

            mParams = params;
            parameter_sanity_check(); // throws exception if bad parameters
            compile(flags);
        }

        OpenCL::~OpenCL()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif

            for (int i = 0; i < programs.size(); i++) {
                delete programs[i];
            }

            clfftTeardown();
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

            // Command queues
            cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            cl::CommandQueue htodqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            cl::CommandQueue dtohqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

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
            int jobsize = mParams.get_job_size_gridder();

            // Start gridder
            runtime -= omp_get_wtime();
            const int nr_streams = 3;
            #pragma omp parallel num_threads(nr_streams)
            {
               // Load kernel functions
                kernel::Gridder kernel_gridder(*programs[which_program[kernel::name_gridder]], mParams);
                kernel::GridFFT kernel_fft(mParams);

                // Events
                vector<cl::Event> inputReady(1), computeReady(1), outputReady(1);
                htodqueue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

                // Private device memory
                cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_VISIBILITIES);
                cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_UVW);
                cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_SUBGRIDS);
                cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_METADATA);

                // Performance counters
                vector<PerformanceCounter*> counters_gridder;

                #pragma omp for schedule(dynamic)
                for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // Offsets
                    size_t uvw_offset          = s * nr_timesteps * 3 * sizeof(float);
                    size_t visibilities_offset = s * nr_timesteps * nr_channels * nr_polarizations * sizeof(complex<float>);
                    size_t subgrids_offset     = s * subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>);
                    size_t metadata_offset     = s * 5 * sizeof(int);

                    // Performance counters
                    PerformanceCounter *counter_gridder = new PerformanceCounter("gridder");
                    #if defined(MEASURE_POWER_ARDUINO)
                    counter_gridder->setPowerSensor(powerSensor);
                    #endif
                    counters_gridder.push_back(counter_gridder);

                    #pragma omp critical (GPU)
                    {
    						// Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&computeReady, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, current_jobsize * SIZEOF_UVW, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, current_jobsize * SIZEOF_VISIBILITIES, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, current_jobsize * SIZEOF_METADATA, NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

    						// Create FFT plan
                            kernel_fft.plan(context, subgridsize, current_jobsize);

    						// Launch gridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            kernel_gridder.launchAsync(executequeue, current_jobsize, w_offset, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, *counters_gridder.back());

    						// Launch FFT
                            kernel_fft.launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

    						// Copy subgrid to host
                            dtohqueue.enqueueMarkerWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_subgrids, h_subgrids, 0, subgrids_offset, current_jobsize * SIZEOF_SUBGRIDS, NULL, &outputReady[0]);
                    }
                }

                // Wait for device to host transfer to finish
                outputReady[0].wait();
            }
            runtime += omp_get_wtime();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            auxiliary::report("|gridding", runtime, 0, 0, 0);
            auxiliary::report_visibilities("|gridding", runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
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

            // Command queue
            cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            cl::CommandQueue htodqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            cl::CommandQueue dtohqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

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
            const int jobsize = mParams.get_job_size_degridder();

            // Start gridder
            runtime -= omp_get_wtime();
            const int nr_streams = 3;
            #pragma omp parallel num_threads(nr_streams)
            {
                // Load kernel functions
                kernel::Degridder kernel_degridder(*programs[which_program[kernel::name_degridder]], mParams);
                kernel::GridFFT kernel_fft(mParams);

                // Events
                vector<cl::Event> inputReady(1), computeReady(1), outputReady(1);
                htodqueue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

                // Private device memory
                cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_VISIBILITIES);
                cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_UVW);
                cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_WRITE_ONLY, jobsize * SIZEOF_SUBGRIDS);
                cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_METADATA);

                // Performance counters
                vector<PerformanceCounter*> counters_degridder;

                #pragma omp for schedule(dynamic)
                for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // Offsets
                    size_t uvw_offset          = s * nr_timesteps * 3 * sizeof(float);
                    size_t visibilities_offset = s * nr_timesteps * nr_channels * nr_polarizations * sizeof(complex<float>);
                    size_t subgrids_offset     = s * subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>);
                    size_t metadata_offset     = s * 5 * sizeof(int);

                    // Performance counters
                    PerformanceCounter *counter_degridder = new PerformanceCounter("degridder");
                    #if defined(MEASURE_POWER_ARDUINO)
                    counter_degridder->setPowerSensor(powerSensor);
                    #endif
                    counters_degridder.push_back(counter_degridder);


                    #pragma omp critical (GPU)
                    {
    						// Copy input data to device
                            htodqueue.enqueueMarkerWithWaitList(&computeReady, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, current_jobsize * SIZEOF_UVW, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, current_jobsize * SIZEOF_METADATA, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_subgrids, d_subgrids, subgrids_offset, 0, current_jobsize * SIZEOF_SUBGRIDS, NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

    						// Create FFT plan
                            kernel_fft.plan(context, subgridsize, current_jobsize);

    						// Launch FFT
                            executequeue.enqueueBarrierWithWaitList(&inputReady, NULL);
                            kernel_fft.launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD);

    						// Launch degridder kernel
                            kernel_degridder.launchAsync(executequeue, current_jobsize, w_offset, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, *counters_degridder.back());
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

    						// Copy visibilities to host
                            dtohqueue.enqueueBarrierWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, current_jobsize * SIZEOF_VISIBILITIES, NULL, &outputReady[0]);
                    }
                }

                // Wait for device to host transfer to finish
                outputReady[0].wait();
            }
            runtime += omp_get_wtime();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            auxiliary::report("|degridding", runtime, 0, 0, 0);
            auxiliary::report_visibilities("|degridding", runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
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
            stringstream _srcdir;
            _srcdir << string(IDG_SOURCE_DIR);
            _srcdir << "/src/OpenCL/Reference/kernels";
            string srcdir = _srcdir.str();

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

            // Build parameters tring
            stringstream _parameters;
            _parameters << " " << flags;
            _parameters << " " << "-I " << srcdir;
            _parameters << " " << mparameters;
            string parameters = _parameters.str();

            // Create vector of devices
            std::vector<cl::Device> devices;
            devices.push_back(device);

            // Add all kernels to build
            vector<string> v;
            v.push_back("KernelGridder.cl");
            v.push_back("KernelDegridder.cl");

            // Build OpenCL programs
            for (int i = 0; i < v.size(); i++) {
                // Get source filename
                stringstream _source_file_name;
                _source_file_name << srcdir << "/" << v[i];
                string source_file_name = _source_file_name.str();

                // Read source from file
                ifstream source_file(source_file_name.c_str());
                string source(std::istreambuf_iterator<char>(source_file),
                             (std::istreambuf_iterator<char>()));
                source_file.close();

                // Create OpenCL program
                cl::Program *program = new cl::Program(context, source);
                try {
                    // Build the program
                    (*program).build(devices, parameters.c_str());
                    programs.push_back(program);

                    // Print information about compilation
                    std::string msg;
                    (*program).getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
                    cout << "Compiling " << _source_file_name.str() << ":"
                         << endl << parameters << endl << msg;
                } catch (cl::Error &error) {
                    if (strcmp(error.what(), "clBuildProgram") == 0) {
                        // Print error message
                        std::string msg;
                        (*program).getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
                        std::cerr << msg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
            } // for each library

            // Fill which_program structure
            which_program[kernel::name_gridder] = 0;
            which_program[kernel::name_degridder] = 1;
        } // compile

        void OpenCL::parameter_sanity_check()
        {
            #if defined(DEBUG)
            cout << "OpenCL::" << __func__ << endl;
            #endif
        }

    } // namespace proxy

} // namespace idg
