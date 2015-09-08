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


        void CUDA::grid_onto_subgrids(CU_GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            run_gridder(CU_GRIDDER_ARGUMENTS);
        }


        void CUDA::add_subgrids_to_grid(CU_ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            run_adder(CU_ADDER_ARGUMENTS);
        }


        void CUDA::split_grid_into_subgrids(CU_SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            run_splitter(CU_SPLITTER_ARGUMENTS);
        }


        void CUDA::degrid_from_subgrids(CU_DEGRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            run_degridder(CU_DEGRIDDER_ARGUMENTS);
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
        void CUDA::run_gridder(CU_GRIDDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif

            // Performance measurements
            double runtime, runtime_gridder, runtime_fft;
            double total_runtime_gridder = 0;
            double total_runtime_fft = 0;

            // Constants
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_channels = mParams.get_nr_channels();
            auto nr_polarizations = mParams.get_nr_polarizations();
            auto subgridsize = mParams.get_subgrid_size();

            // load kernel functions
            kernel::Gridder kernel_gridder(*(modules[which_module[kernel::name_gridder]]), mParams);
            kernel::GridFFT kernel_fft(mParams);

            // Initialize
            cu::Stream executestream;
            cu::Stream htodstream;
            cu::Stream dtohstream;
            const int nr_streams = 2;

            // Set jobsize to match available gpu memory
            uint64_t device_memory_required = SIZEOF_VISIBILITIES + SIZEOF_UVW + SIZEOF_SUBGRIDS + SIZEOF_METADATA;
            uint64_t device_memory_available = device.free_memory();
            int jobsize = (device_memory_available * 0.7) / (device_memory_required * nr_streams);

            // Make sure that jobsize isn't too large
            int max_jobsize = nr_subgrids / 2;
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

            // Start gridder
            #pragma omp parallel num_threads(nr_streams)
            {
                // Initialize
	            context.setCurrent();
	            cu::Event inputFree;
                cu::Event outputFree;
                cu::Event inputReady;
                cu::Event outputReady;
                int thread_num = omp_get_thread_num();
                int current_jobsize = jobsize;

        	    // Private device memory
            	cu::DeviceMemory d_visibilities(jobsize * SIZEOF_VISIBILITIES);
            	cu::DeviceMemory d_uvw(jobsize * SIZEOF_UVW);
        	    cu::DeviceMemory d_subgrids(jobsize * SIZEOF_SUBGRIDS);
                cu::DeviceMemory d_metadata(jobsize * SIZEOF_METADATA);
    
                for (unsigned int s = thread_num * jobsize; s < nr_subgrids; s += nr_streams * jobsize) {
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

    	            // Copy input data to device
                    #pragma omp critical
                    {
                        htodstream.waitEvent(inputFree);
                        htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, current_jobsize * SIZEOF_VISIBILITIES);
                        htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, current_jobsize * SIZEOF_UVW);
                        htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, current_jobsize * SIZEOF_METADATA);
                        htodstream.record(inputReady);
                    }

                    // Create FFT plan
                    kernel_fft.plan(subgridsize, current_jobsize);

    	            #pragma omp critical
                    {
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
                    }
	            
    	            #pragma omp critical 
                    {
                        // Copy subgrid to host
    	                dtohstream.waitEvent(outputReady);
    	                dtohstream.memcpyDtoHAsync(subgrids_ptr, d_subgrids, current_jobsize * SIZEOF_SUBGRIDS);
                        dtohstream.record(outputFree);
                    }
                } // end for s

                // Wait for final transfer to finish 
                dtohstream.synchronize();
            }

            runtime += omp_get_wtime();

            #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
            clog << endl;
            clog << "Total: gridding" << endl;
            auxiliary::report_visibilities(runtime, nr_baselines, nr_timesteps * nr_timeslots, nr_channels);
            clog << endl;
            #endif


       } // run_gridder



        void CUDA::run_adder(CU_ADDER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_adder


        void CUDA::run_splitter(CU_SPLITTER_PARAMETERS)
        {
            #if defined(DEBUG)
            cout << "CUDA::" << __func__ << endl;
            #endif
        } // run_splitter


        void CUDA::run_degridder(CU_DEGRIDDER_PARAMETERS)
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
