#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "idg-config.h"
#include "Maxwell.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

#if defined(MEASURE_POWER)
#define _QUOTE(str) #str
#define QUOTE(str) _QUOTE(str)
#define STR_POWER_SENSOR QUOTE(POWER_SENSOR)
#define STR_POWER_FILE QUOTE(POWER_FILE)
#endif

using namespace std;

namespace idg {
    namespace proxy {
        namespace cuda {
            // Power sensor
            static PowerSensor *powerSensor;
     

            /// Constructors
            Maxwell::Maxwell(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                #if defined(MEASURE_POWER)
                cout << "Opening power sensor: " << STR_POWER_SENSOR << endl;
                cout << "Writing power consumption to file: " << STR_POWER_FILE << endl;
                powerSensor = new PowerSensor(STR_POWER_SENSOR, STR_POWER_FILE);
                #endif
            }

            /*
                Size of data structures for a single job
            */
            #define SIZEOF_SUBGRIDS 1ULL * nr_polarizations * subgridsize * subgridsize * sizeof(complex<float>)
            #define SIZEOF_UVW      1ULL * nr_timesteps * 3 * sizeof(float)
            #define SIZEOF_VISIBILITIES 1ULL * nr_timesteps * nr_channels * nr_polarizations * sizeof(complex<float>)
            #define SIZEOF_METADATA 1ULL * 5 * sizeof(int)
            #define SIZEOF_GRID     1ULL * nr_polarizations * gridsize * gridsize * sizeof(complex<float>)

            /// Low level routines
            void Maxwell::run_gridder(CU_GRIDDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                #endif
    
                // Performance measurements
                double runtime = 0;
    
                // Constants
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
    
                // load kernel functions
                kernel::Gridder kernel_gridder(*(modules[which_module[kernel::name_gridder]]), mParams);
    
                // Initialize
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
                const int nr_streams = 3;
    
                // Set jobsize to match available gpu memory
                uint64_t device_memory_required = SIZEOF_VISIBILITIES + SIZEOF_UVW + SIZEOF_SUBGRIDS + SIZEOF_METADATA;
                uint64_t device_memory_available = device.free_memory();
                int jobsize = (device_memory_available * 0.7) / (device_memory_required * nr_streams);
    
                // Make sure that jobsize isn't too large
                int max_jobsize = nr_subgrids / 8;
                if (jobsize >= max_jobsize) {
                    jobsize = max_jobsize;
                }
    
                #if defined (DEBUG)
                clog << "nr_subgrids: " << nr_subgrids << endl;
                clog << "jobsize:     " << jobsize << endl;
                clog << "free size:   " << device_memory_available * 1e-9 << " Gb" << endl;
                clog << "buffersize:  " << nr_streams * jobsize * device_memory_required * 1e-9 << " Gb" << endl;
                #endif
    
     	        runtime = -omp_get_wtime();
                #if defined(MEASURE_POWER)
                PowerSensor::State startState = powerSensor->read();
                #endif
    
                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
    	            context.setCurrent();
    	            cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    //int thread_num = omp_get_thread_num();
                    //int current_jobsize = jobsize;
                    kernel::GridFFT kernel_fft(mParams);
    
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
    						kernel_gridder.launchAsync(
    							executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
    							d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
    						 
    						// Launch FFT
    						kernel_fft.launchAsync(executestream, d_subgrids, CUFFT_INVERSE);
    						executestream.record(outputReady);
    						executestream.record(inputFree);
    
    						// Copy subgrid to host
    						dtohstream.waitEvent(outputReady);
    						dtohstream.memcpyDtoHAsync(subgrids_ptr, d_subgrids, current_jobsize * SIZEOF_SUBGRIDS);
    						dtohstream.record(outputFree);
    					}
    
    					outputFree.synchronize();
                    } // end for s
                }
    
                runtime += omp_get_wtime();
                #if defined(MEASURE_POWER)
                PowerSensor::State stopState = powerSensor->read();
                #endif
    
                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                clog << "Total: gridding" << endl;
                clog << "Runtime: " << runtime << " s" << endl;
                auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                #if defined(MEASURE_POWER)
                auxiliary::report_power(PowerSensor::seconds(startState, stopState),
                                        PowerSensor::Watt(startState, stopState),
                                        PowerSensor::Joules(startState, stopState));
                #endif
                clog << endl;
                #endif
            } // run_gridder

            void Maxwell::run_degridder(CU_DEGRIDDER_PARAMETERS)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                #endif
    
                // Performance measurements
                double runtime = 0;
    
                // Constants
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
    
                // load kernel
                kernel::Degridder kernel_degridder(*(modules[which_module[kernel::name_degridder]]), mParams);
    
                // Initialize
                const int nr_streams = 3;
                cu::Stream executestream;
                cu::Stream htodstream;
                cu::Stream dtohstream;
    
                // Set jobsize to match available gpu memory
                uint64_t device_memory_required = SIZEOF_VISIBILITIES + SIZEOF_UVW + SIZEOF_SUBGRIDS + SIZEOF_METADATA;
                uint64_t device_memory_available = device.free_memory();
                int jobsize = (device_memory_available * 0.7) / (device_memory_required * nr_streams);
    
                // Make sure that jobsize isn't too large
                int max_jobsize = nr_subgrids / 8;
                if (jobsize >= max_jobsize) {
                    jobsize = max_jobsize;
                }
    
     	        runtime = -omp_get_wtime();
    
                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Initialize
    	            context.setCurrent();
    	            cu::Event inputFree;
                    cu::Event outputFree;
                    cu::Event inputReady;
                    cu::Event outputReady;
                    //int thread_num = omp_get_thread_num();
                    int current_jobsize = jobsize;
                    kernel::GridFFT kernel_fft(mParams);
    
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
    						kernel_fft.launchAsync(executestream, d_subgrids, CUFFT_FORWARD);
    
    						// Launch degridder kernel
    						executestream.waitEvent(outputFree);
    						kernel_degridder.launchAsync(
    							executestream, current_jobsize, w_offset, d_uvw, d_wavenumbers,
    							d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids);
    						 
    						executestream.record(outputReady);
    						executestream.record(inputFree);
    				
    						// Copy visibilities to host
    						dtohstream.waitEvent(outputReady);
    						dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, current_jobsize * SIZEOF_VISIBILITIES);
    						dtohstream.record(outputFree);
    					}
    
    					outputFree.synchronize();
                    } // end for s
                }
    
                runtime += omp_get_wtime();
    
                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                clog << "Total: degridding" << endl;
                clog << "Runtime: " << runtime << " s" << endl;
                auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
                clog << endl;
                #endif
            } // run_degridder



            ProxyInfo Maxwell::default_info() {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif
                
                string srcdir = string(IDG_SOURCE_DIR) 
                    + "/src/CUDA/Maxwell/kernels";
                
                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif
                
                // Create temp directory
                string tmpdir = make_tempdir();
                
                // Create proxy info
                ProxyInfo p = default_proxyinfo(srcdir, tmpdir);
                
                return p;

               return CUDA::default_info();
           }

           string Maxwell::default_compiler() {
                return CUDA::default_compiler();
           }

           string Maxwell::default_compiler_flags() {
               return CUDA::default_compiler_flags();

           }

        } // namespace cuda
    } // namespace proxy
} // namespace idg
