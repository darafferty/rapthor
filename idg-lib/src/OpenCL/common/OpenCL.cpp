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

using namespace std;

namespace idg {
    namespace proxy {
        namespace opencl {
            /// Constructors
            OpenCL::OpenCL(
                Parameters params,
                cl::Context &context,
                unsigned deviceNumber,
                Compilerflags flags) :
                context(context)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            	// Get devices
            	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
                device = devices[deviceNumber];

                // Set/check parameters
                mParams = params;
                parameter_sanity_check(); // throws exception if bad parameters

                // Compile kernels
                compile(flags);

                // Initialize clFFT
                clfftSetupData setup;
                clfftSetup(&setup);

                // Initialize power sensor
                #if defined(MEASURE_POWER_ARDUINO)
                const char *str_power_sensor = getenv("POWER_SENSOR");
                if (!str_power_sensor) str_power_sensor = POWER_SENSOR;
                const char *str_power_file = getenv("POWER_FILE");
                if (!str_power_file) str_power_file = POWER_FILE;
                cout << "Opening power sensor: " << str_power_sensor << endl;
                cout << "Writing power consumption to file: " << str_power_file << endl;
                powerSensor.init(str_power_sensor, str_power_file);
                #endif
            }

            OpenCL::~OpenCL()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (int i = 0; i < programs.size(); i++) {
                    delete programs[i];
                }

                clfftTeardown();
            }

            string OpenCL::default_compiler_flags() {
                return "-cl-fast-relaxed-math";
            }

            /* High level routines */
            void OpenCL::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

#if 0
                // Command queues
                cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue htodqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE)
                cl::CommandQueue dtohqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE)

                // Performance measurements
                double runtime = 0;

                // Constants auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                // Set jobsize
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
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_ONLY, jobsize * SIZEOF_VISIBILITIES);
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_ONLY, jobsize * SIZEOF_UVW);
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_SUBGRIDS);
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_ONLY, jobsize * SIZEOF_METADATA);

                    // Performance counters
                    PerformanceCounter counter_gridder;
                    PerformanceCounter counter_fft;
                    #if defined(MEASURE_POWER_ARDUINO)
                    counter_gridder.setPowerSensor(&powerSensor);
                    counter_fft.setPowerSensor(&powerSensor);
                    #endif

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
                            htodqueue.enqueueMarkerWithWaitList(&computeReady, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, current_jobsize * SIZEOF_UVW, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, current_jobsize * SIZEOF_VISIBILITIES, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, current_jobsize * SIZEOF_METADATA, NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

        					// Create FFT plan
                            kernel_fft.plan(context, executequeue, subgridsize, current_jobsize * nr_polarizations);

        					// Launch gridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            kernel_gridder.launchAsync(executequeue, current_jobsize, w_offset, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counter_gridder);

        					// Launch FFT
                            kernel_fft.launchAsync(executequeue, d_subgrids, CLFFT_BACKWARD, counter_fft);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

        					// Copy subgrid to host
                            dtohqueue.enqueueMarkerWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_subgrids, h_subgrids, 0, subgrids_offset, current_jobsize * SIZEOF_SUBGRIDS, NULL, &outputReady[0]);
                        }

                        // Wait for device to host transfer to finish
                        outputReady[0].wait();
                    }
                }
                runtime += omp_get_wtime();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auxiliary::report("|gridding", runtime, 0, 0, 0);
                auxiliary::report_visibilities("|gridding", runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
#endif
            } // grid_visibilities


            void OpenCL::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
               #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
#if 0
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
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_VISIBILITIES);
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_UVW);
                    cl::Buffer d_subgrids     = cl::Buffer(context, CL_MEM_READ_WRITE, jobsize * SIZEOF_SUBGRIDS);
                    cl::Buffer d_metadata     = cl::Buffer(context, CL_MEM_READ_ONLY,  jobsize * SIZEOF_METADATA);

                    // Performance counters
                    PerformanceCounter counter_degridder;
                    PerformanceCounter counter_fft;
                    #if defined(MEASURE_POWER_ARDUINO)
                    counter_degridder.setPowerSensor(&powerSensor);
                    counter_fft.setPowerSensor(&powerSensor);
                    #endif

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
                            htodqueue.enqueueMarkerWithWaitList(&computeReady, NULL);
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, current_jobsize * SIZEOF_UVW, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, current_jobsize * SIZEOF_METADATA, NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_subgrids, d_subgrids, subgrids_offset, 0, current_jobsize * SIZEOF_SUBGRIDS, NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

        					// Create FFT plan
                            kernel_fft.plan(context, executequeue, subgridsize, current_jobsize * nr_polarizations);

        					// Launch FFT
                            executequeue.enqueueBarrierWithWaitList(&inputReady, NULL);
                            kernel_fft.launchAsync(executequeue, d_subgrids, CLFFT_FORWARD, counter_fft);

        					// Launch degridder kernel
                            kernel_degridder.launchAsync(executequeue, current_jobsize, w_offset, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counter_degridder);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

        					// Copy visibilities to host
                            dtohqueue.enqueueBarrierWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, current_jobsize * SIZEOF_VISIBILITIES, NULL, &outputReady[0]);
                        }

                        // Wait for device to host transfer to finish
                        outputReady[0].wait();
                    }
                }
                runtime += omp_get_wtime();

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                auxiliary::report("|degridding", runtime, 0, 0, 0);
                auxiliary::report_visibilities("|degridding", runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
#endif
            } // degrid_visibilities


            void OpenCL::transform(
                DomainAtoDomainB direction,
                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

#if 0
                // Constants
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();

                // Command queue
                cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

                // Events
                vector<cl::Event> outputReady(1);

                // Device memory
                cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_WRITE, SIZEOF_GRID);

                // Performance counter
                PerformanceCounter counter_fft;
                #if defined(MEASURE_POWER_ARDUINO)
                counter_fft.setPowerSensor(&powerSensor);
                #endif

                // Load kernel function
                kernel::GridFFT kernel_fft(mParams);

                // Copy grid to device
                queue.enqueueCopyBuffer(h_grid, d_grid, 0, 0, SIZEOF_GRID, NULL, NULL);

                // Create FFT plan
                kernel_fft.plan(context, queue, gridsize, nr_polarizations);

        		// Launch FFT
                kernel_fft.launchAsync(queue, d_grid, direction, counter_fft);

                // Copy grid to host
                queue.enqueueCopyBuffer(d_grid, h_grid, 0, 0, SIZEOF_GRID, NULL, &outputReady[0]);

                // Wait for fft to finish
                outputReady[0].wait();
                clog << endl;
#endif
            } // transform

            void OpenCL::compile(Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Source directory
                stringstream _srcdir;
                _srcdir << string(IDG_INSTALL_DIR);
                _srcdir << "/lib/kernels/OpenCL";
                string srcdir = _srcdir.str();

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
                string mparameters = Parameters::definitions(
                    mParams.get_nr_stations(),
                    mParams.get_nr_baselines(),
                    mParams.get_nr_channels(),
                    mParams.get_nr_time(),
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
                //v.push_back("KernelDegridder.cl");

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

                    // Print information about compilation
                    cout << "Compiling " << _source_file_name.str() << ":"
                         << endl << parameters << endl;

                    // Create OpenCL program
                    cl::Program *program = new cl::Program(context, source);
                    try {
                        // Build the program
                        (*program).build(devices, parameters.c_str());
                        programs.push_back(program);

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
                //which_program[kernel::name_degridder] = 1;
            } // compile

            void OpenCL::parameter_sanity_check()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }
        } // namespace opencl
    } // namespace proxy
} // namespace idg
