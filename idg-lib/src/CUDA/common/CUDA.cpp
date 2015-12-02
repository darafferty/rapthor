#include "CUDA.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cuda {
            /*
                Power measurement
            */
            static PowerSensor *powerSensor;

            class PowerRecord {
                public:
                    void enqueue(cu::Stream &stream);
                    static void getPower(CUstream, CUresult, void *userData);
                    PowerSensor::State state;
                    cu::Event event;
            };

            void PowerRecord::enqueue(cu::Stream &stream) {
                stream.record(event);
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &state);
            }

            void PowerRecord::getPower(CUstream, CUresult, void *userData) {
                *static_cast<PowerSensor::State *>(userData) = powerSensor->read();
            }

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

                #if defined(MEASURE_POWER_ARDUINO)
                const char *str_power_sensor = getenv("POWER_SENSOR");
                if (!str_power_sensor) str_power_sensor = STR_POWER_SENSOR;
                const char *str_power_file = getenv("POWER_FILE");
                if (!str_power_file) str_power_file = STR_POWER_FILE;
                cout << "Opening power sensor: " << str_power_sensor << endl;
                cout << "Writing power consumption to file: " << str_power_file << endl;
                powerSensor = new PowerSensor(str_power_sensor, str_power_file);
                #else
                powerSensor = new PowerSensor();
                #endif
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

            string CUDA::default_compiler() {
                return "nvcc";
            }

            string CUDA::default_compiler_flags() {
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

                int sign = (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;
                cout << "Not implemented" << endl;
            }


            void CUDA::grid_onto_subgrids(CU_GRIDDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif

                // Constants
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                kernel::Gridder kernel_gridder(*(modules[which_module[kernel::name_gridder]]), mParams);
                cu::Module *module_fft = (modules[which_module[kernel::name_fft]]);

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Performance measurements
                double total_runtime_gridding = 0;
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                total_runtime_gridding = -omp_get_wtime();
                PowerSensor::State startState = powerSensor->read();

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
                    context.setCurrent();
                    cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    kernel::GridFFT kernel_fft(*module_fft, mParams);

                    // Private device memory
                	cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                	cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        int current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + s * visibilities_elements;
                        void *subgrids_ptr     = (complex<float>*) h_subgrids + s * subgrid_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[3];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
                		{
                			// Copy input data to device
                			htodstream.waitEvent(inputFree);
                			htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, current_jobsize * SIZEOF_VISIBILITIES);
                			htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, current_jobsize * SIZEOF_UVW);
                			htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                			htodstream.record(inputReady);

                			// Create FFT plan
                			kernel_fft.plan(subgridsize, current_jobsize);

                			// Launch gridder kernel
                			executestream.waitEvent(inputReady);
                			executestream.waitEvent(outputFree);
                            powerRecords[0].enqueue(executestream);
                			kernel_gridder.launchAsync(
                				executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
                				d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[1].enqueue(executestream);

                			// Launch FFT
                			kernel_fft.launchAsync(executestream, d_subgrids, CUFFT_INVERSE);
                            powerRecords[2].enqueue(executestream);
                			executestream.record(outputReady);
                			executestream.record(inputFree);

                			// Copy subgrid to host
                			dtohstream.waitEvent(outputReady);
                			dtohstream.memcpyDtoHAsync(subgrids_ptr, d_subgrids, current_jobsize * SIZEOF_SUBGRIDS);
                			dtohstream.record(outputFree);
                		}

                		outputFree.synchronize();

                        double runtime_gridder = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_fft     = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                                                     kernel_gridder.flops(current_jobsize),
                                                     kernel_gridder.bytes(current_jobsize),
                                                     PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("    fft", runtime_fft,
                                                     kernel_fft.flops(subgridsize, current_jobsize),
                                                     kernel_fft.bytes(subgridsize, current_jobsize),
                                                     PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_gridder += runtime_gridder;
                        total_runtime_fft     += runtime_fft;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_gridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor->read();
                kernel::GridFFT kernel_fft(*module_fft, mParams);
                uint64_t total_flops_gridder  = kernel_gridder.flops(nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder.bytes(nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft.flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft.bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft;
                double   total_watt_gridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
            }


            void CUDA::add_subgrids_to_grid(CU_ADDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                cout << "Not implemented" << endl;
                #endif
            }


            void CUDA::split_grid_into_subgrids(CU_SPLITTER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                cout << "Not implemented" << endl;
                #endif
            }


            void CUDA::degrid_from_subgrids(CU_DEGRIDDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif

                // Constants
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size();

                // load kernel
                kernel::Degridder kernel_degridder(*(modules[which_module[kernel::name_degridder]]), mParams);
                cu::Module *module_fft = (modules[which_module[kernel::name_fft]]);

                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
         	    total_runtime_degridding = -omp_get_wtime();
                PowerSensor::State startState = powerSensor->read();

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
        	        context.setCurrent();
        	        cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    int current_jobsize = jobsize;
                    kernel::GridFFT kernel_fft(*module_fft, mParams);

                	// Private device memory
                    cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
                    cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
                	cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                    cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);

                    #pragma omp for schedule(dynamic)
                    for (unsigned s = 0; s < nr_subgrids; s += jobsize) {
                        // Prevent overflow
                        current_jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_timesteps * 3;
                        int visibilities_elements = nr_timesteps * nr_channels * nr_polarizations;
                        int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;
                        int metadata_elements     = 5;

                        // Pointers to data for current batch
                        void *uvw_ptr          = (float *) h_uvw + s * uvw_elements;
                        void *visibilities_ptr = (complex<float>*) h_visibilities + s * visibilities_elements;
                        void *subgrids_ptr     = (complex<float>*) h_subgrids + s * subgrid_elements;
                        void *metadata_ptr     = (int *) h_metadata + s * metadata_elements;

                        // Power measurement
                        PowerRecord powerRecords[3];

                        #pragma omp critical (GPU) // TODO: use multiple locks for multiple GPUs
        				{
        					// Copy input data to device
        					htodstream.waitEvent(inputFree);
        					htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, current_jobsize * SIZEOF_UVW);
        					htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, current_jobsize * SIZEOF_SUBGRIDS);
        					htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
        					htodstream.record(inputReady);

        					// Create FFT plan
        					kernel_fft.plan(subgridsize, current_jobsize);

        					// Launch FFT
        					executestream.waitEvent(inputReady);
                            powerRecords[0].enqueue(executestream);
        					kernel_fft.launchAsync(executestream, d_subgrids, CUFFT_FORWARD);
                            powerRecords[1].enqueue(executestream);

        					// Launch degridder kernel
        					executestream.waitEvent(outputFree);
        					kernel_degridder.launchAsync(
        						executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
        						d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
                            powerRecords[2].enqueue(executestream);
        					executestream.record(outputReady);
        					executestream.record(inputFree);

        					// Copy visibilities to host
        					dtohstream.waitEvent(outputReady);
        					dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
        					dtohstream.record(outputFree);
        				}

        				outputFree.synchronize();
                        double runtime_fft       = PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state);
                        double runtime_degridder = PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state);
                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("      fft", PowerSensor::seconds(powerRecords[0].state, powerRecords[1].state),
                                                       kernel_fft.flops(subgridsize, current_jobsize),
                                                       kernel_fft.bytes(subgridsize, current_jobsize),
                                                       PowerSensor::Watt(powerRecords[0].state, powerRecords[1].state));
                        auxiliary::report("degridder", PowerSensor::seconds(powerRecords[1].state, powerRecords[2].state),
                                                       kernel_degridder.flops(current_jobsize),
                                                       kernel_degridder.bytes(current_jobsize),
                                                       PowerSensor::Watt(powerRecords[1].state, powerRecords[2].state));
                        #endif
                        #if defined(REPORT_TOTAL)
                        total_runtime_degridder += runtime_degridder;
                        total_runtime_fft       += runtime_fft;
                        #endif
                    } // end for s
                }

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_degridding += omp_get_wtime();
                PowerSensor::State stopState = powerSensor->read();
                kernel::GridFFT kernel_fft(*module_fft, mParams);
                uint64_t total_flops_fft        = kernel_fft.flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft.bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_degridder  = kernel_degridder.flops(nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder.bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                double   total_watt_degridding  = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                 clog << endl;
                #endif
            }


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
            } // compile

            void CUDA::parameter_sanity_check() {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif
            }


            void CUDA::load_shared_objects() {
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
		}
    } // namespace proxy
} // namespace idg
