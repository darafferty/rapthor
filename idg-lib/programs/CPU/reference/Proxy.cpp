#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime

#include "Proxy.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {
  
  namespace proxy {
    
    /// Constructors
    CPU::CPU(Compiler compiler, 
             Compilerflags flags,
             Parameters params,
             AlgorithmParameters algparams,
             ProxyInfo info) 
      : mParams(params),
        mAlgParams(algparams),
        mInfo(info)
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      cout << "Compiler: " << compiler << endl;
      cout << "Compiler flags: " << flags << endl;
      cout << params;
      cout << algparams;
      #endif

      parameter_sanity_check(); // throws exception if bad parameters
      
      compile(compiler, flags); 
      
      load_shared_objects();
      
      find_kernel_functions();
    }
    

    CPU::CPU(CompilerEnvironment cc, 
             Parameters params,
             AlgorithmParameters algparams,
             ProxyInfo info) 
      : mParams(params),
        mAlgParams(algparams),
        mInfo(info)
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif
      
      // find out which compiler to use
      // call CPU(compiler, flags, params, algparams)
      
      cerr << "Constructor not implemented yet" << endl;
    } 
    

    CPU::~CPU() 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // unload shared objects by ~Module
      for (unsigned int i=0; i<modules.size(); i++)
        delete modules[i];
      
      // Delete .so files 
      if( mInfo.delete_shared_objects() ) {
        for (auto libname : mInfo.get_lib_names()) {
          string lib = mInfo.get_path_to_lib() + "/" + libname;
          remove(lib.c_str());
        } 
      }
      
    }


    AlgorithmParameters CPU::default_algparams() 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      AlgorithmParameters p;
      p.set_job_size(128);    // please set sensible value here
      p.set_subgrid_size(32); // please set sensible value here
      p.set_chunk_size(32);   // please set sensible value here
      
      return p;
    }


    ProxyInfo CPU::default_info() 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      ProxyInfo p;

      p.set_path_to_src("../../../kernels/CPU/reference");
      p.set_path_to_lib("../../../lib"); // change to use tmp dir by default

      srand(time(NULL));
      string rnd_str = to_string( rand() );

      string libgridder = "Gridder" + rnd_str + ".so";
      string libdegridder = "Degridder" + rnd_str + ".so";
      string libfft = "FFT" + rnd_str + ".so";
      string libadder = "Adder" + rnd_str + ".so";
      string libsplitter = "Splitter" + rnd_str + ".so";

      p.add_lib(libgridder); 
      p.add_lib(libdegridder);
      p.add_lib(libfft);
      p.add_lib(libadder);
      p.add_lib(libsplitter);
      
      p.add_src_file_to_lib(libgridder, "KernelGridder.cpp");
      p.add_src_file_to_lib(libdegridder, "KernelDegridder.cpp");
      p.add_src_file_to_lib(libfft, "KernelFFT.cpp");
      p.add_src_file_to_lib(libadder, "KernelAdder.cpp");
      p.add_src_file_to_lib(libsplitter, "KernelSplitter.cpp");

      p.set_delete_shared_objects(true);
      
      return p;
    }
    

    /// Methods
    void CPU::grid_visibilities(void *visibilities, 
				void *uvw, 
				void *wavenumbers,
				void *aterm, 
				void *spheroidal, 
				void *baselines, 
				void *grid) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // get parameters
      unsigned int nr_baselines = mParams.get_nr_baselines();
      unsigned int nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      unsigned int subgridsize = mAlgParams.get_subgrid_size();
      unsigned int nr_polarizations = mParams.get_nr_polarizations();

      // allocate subgrids: two different versions dependingon layout?
      size_t size_subgrids = (size_t) nr_baselines*nr_chunks*subgridsize
	                              *subgridsize*nr_polarizations;
      auto subgrids = new complex<float>[size_subgrids];

      // Get job sizes for gridding and adding routines
      int jobsize_gridder = mAlgParams.get_job_size_gridder();
      int jobsize_adder = mAlgParams.get_job_size_adder();

      grid_onto_subgrids(jobsize_gridder, visibilities, uvw, wavenumbers, aterm, 
			 spheroidal, baselines, subgrids);

      add_subgrids_to_grid(jobsize_adder, uvw, subgrids, grid); 

      // free subgrid
      delete[] subgrids;
    }


    void CPU::degrid_visibilities(void *grid,
				  void *uvw,
				  void *wavenumbers, 
				  void *aterm,
				  void *spheroidal, 
				  void *baselines,
				  void *visibilities) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // get parameters
      unsigned int nr_baselines = mParams.get_nr_baselines();
      unsigned int nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      unsigned int subgridsize = mAlgParams.get_subgrid_size();
      unsigned int nr_polarizations = mParams.get_nr_polarizations();

      // allocate subgrids: two different versions dependingon layout?
      size_t size_subgrids = (size_t) nr_baselines*nr_chunks*subgridsize*subgridsize*nr_polarizations;
      auto subgrids = new complex<float>[size_subgrids];

      // Get job sizes for gridding and adding routines
      int jobsize_splitter = mAlgParams.get_job_size_splitter();
      int jobsize_degridder = mAlgParams.get_job_size_degridder();

      split_grid_into_subgrids(jobsize_splitter, uvw, subgrids, grid);

      degrid_from_subgrids(jobsize_degridder, wavenumbers, aterm, baselines, 
			   visibilities, uvw, spheroidal, subgrids); 
      
      // free subgrids
      delete[] subgrids;
    }


    void CPU::transform(DomainAtoDomainB direction, void* grid) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      cout << "Transform direction: " << direction << endl;
      #endif

      int sign = (direction == FourierDomainToImageDomain) ? 0 : 1;

      run_fft(grid, sign);
    }

    
    // lower-level inteface: (de)gridding split into two function calls each
    void CPU::grid_onto_subgrids(int jobsize, void *visibilities, void *uvw, 
				 void *wavenumbers, void *aterm, void *spheroidal, 
				 void *baselines, void *subgrids)
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy
      
      run_gridder(jobsize, visibilities, uvw, wavenumbers, aterm, spheroidal, 
        baselines, subgrids);
    }

    
    void CPU::add_subgrids_to_grid(int jobsize, void *uvw, void *subgrids, 
                                   void *grid) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_adder(jobsize, uvw, subgrids, grid);
    }


    void CPU::split_grid_into_subgrids(int jobsize, void *uvw, void *subgrids, 
				       void *grid)
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_splitter(jobsize, uvw, subgrids, grid);
    }

    
    void CPU::degrid_from_subgrids(int jobsize, void *wavenumbers, void *aterm, 
				   void *baselines, void *visibilities, void *uvw, 
				   void *spheroidal, void *subgrids)
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_degridder(jobsize, wavenumbers, aterm, baselines, visibilities, 
		    uvw, spheroidal, subgrids);
    }


    
    void CPU::run_gridder(int jobsize, void *visibilities, void *uvw, 
			  void *wavenumbers, void *aterm, void *spheroidal, 
			  void *baselines, void *subgrids) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // Performance measurements
      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      double runtime, runtime_gridder, runtime_fft;
      double total_runtime_gridder = 0;
      double total_runtime_fft = 0;
      #endif
      
      // Constants
      auto nr_baselines = mParams.get_nr_baselines();
      auto nr_time = mParams.get_nr_timesteps();
      auto nr_channels = mParams.get_nr_channels();
      auto nr_polarizations = mParams.get_nr_polarizations();
      auto nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      auto subgridsize = mAlgParams.get_subgrid_size();

      // load kernel functions
      kernel::Gridder kernel_gridder(*(modules[which_module[kernel::name_gridder]]));
      kernel::GridFFT kernel_fft(*(modules[which_module[kernel::name_fft]]));

      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime = -omp_get_wtime();
      #endif
      
      // Start gridder
      for (unsigned int bl=0; bl<nr_baselines; bl+=jobsize) {
        // Prevent overflow
        jobsize = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
        
        // Number of elements in batch
        int uvw_elements          = nr_time * 3;
        int visibilities_elements = nr_time * nr_channels * nr_polarizations;
        int subgrid_elements      = nr_chunks * subgridsize * subgridsize * nr_polarizations;
        
        // Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *wavenumbers_ptr  = wavenumbers;
        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
        void *spheroidal_ptr   = spheroidal;
        void *aterm_ptr        = aterm;
        void *subgrids_ptr      = (complex<float>*) subgrids + bl * subgrid_elements;
        void *baselines_ptr    = baselines;

        
        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_gridder = -omp_get_wtime();
        #endif
        
        kernel_gridder.run(jobsize, bl, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
                           spheroidal_ptr, aterm_ptr, baselines_ptr, subgrids_ptr);
        
        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_gridder += omp_get_wtime();
        total_runtime_gridder += runtime_gridder;
        runtime_fft = -omp_get_wtime();
        #endif
        
        #if ORDER == ORDER_BL_V_U_P
        kernel_fft.run(subgridsize, jobsize, subgrids_ptr, FFTW_BACKWARD, FFT_LAYOUT_YXP);
        #elif ORDER == ORDER_BL_P_V_U
        kernel_fft.run(subgridsize, jobsize, subgrids_ptr, FFTW_BACKWARD, FFT_LAYOUT_PYX);
        #endif
        
        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_fft += omp_get_wtime();
        total_runtime_fft += runtime_fft;
        #endif
        
        #if defined(REPORT_VERBOSE)
        auxiliary::report("gridder", runtime_gridder,
                          kernel_gridder.flops(jobsize),
                          kernel_gridder.bytes(jobsize));
        auxiliary::report("fft", runtime_fft,
                          kernel_fft.flops(subgridsize, nr_baselines),
                          kernel_fft.bytes(subgridsize, nr_baselines));
        #endif
        
      } // end for bl
      
      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime += omp_get_wtime();
      clog << endl;
      clog << "Total: gridding" << endl;
      auxiliary::report("gridder", total_runtime_gridder,
                        kernel_gridder.flops(nr_baselines),
                        kernel_gridder.bytes(nr_baselines));
      auxiliary::report("fft", total_runtime_fft,
                        kernel_fft.flops(subgridsize, nr_baselines),
                        kernel_fft.bytes(subgridsize, nr_baselines));
      auxiliary::report_runtime(runtime);
      clog << endl;
      #endif

    } // run_gridder



    void CPU::run_adder(int jobsize, void *uvw, void *subgrids, void *grid) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // Performance measurements
      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      double runtime, runtime_adder;
      double total_runtime_adder = 0;
      #endif

      // Constants
      auto nr_baselines = mParams.get_nr_baselines();
      auto nr_time = mParams.get_nr_timesteps();
      auto nr_polarizations = mParams.get_nr_polarizations();
      auto nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      auto subgridsize = mAlgParams.get_subgrid_size();

      // Load kernel function
      kernel::Adder kernel_adder(*(modules[which_module[kernel::name_adder]]));

      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime = -omp_get_wtime();
      #endif

      // Run adder
      for (unsigned int bl=0; bl<nr_baselines; bl+=jobsize) {
        // Prevent overflow
        jobsize = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
        
        // Number of elements in batch
        int uvw_elements     = nr_time * 3;
        int subgrid_elements = nr_chunks * subgridsize * subgridsize * nr_polarizations;
        
        // Pointer to data for current jobs
        void *uvw_ptr     = (float*) uvw + bl * uvw_elements;
        void *subgrid_ptr = (complex<float>*) subgrids + bl * subgrid_elements;
        void *grid_ptr    = grid;
        
        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_adder = -omp_get_wtime();
        #endif
        
        kernel_adder.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);
        
        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_adder += omp_get_wtime();
        total_runtime_adder += runtime_adder;
        #endif
        
        #if defined(REPORT_VERBOSE)
        auxiliary::report("adder", runtime_adder,
                          kernel_adder.flops(jobsize),
                          kernel_adder.bytes(jobsize));
        #endif
        
      } // end for bl
      
      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime += omp_get_wtime();
      clog << endl;
      clog << "Total: adding" << endl;
      auxiliary::report("adder", total_runtime_adder,
                        kernel_adder.flops(nr_baselines),
                        kernel_adder.bytes(nr_baselines));
      auxiliary::report_runtime(runtime);
      auxiliary::report_subgrids(runtime, nr_baselines);
      clog << endl;
      #endif
      
    } // run_adder



    void CPU::run_splitter(int jobsize, void *uvw, void *subgrids, void *grid) 
    {
      #if defined(DEBUG) 
      cout << "CPU::" << __func__ << endl;
      #endif

      // Performance measurements
      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      double runtime, runtime_splitter;
      double total_runtime_splitter = 0;
      #endif

      // Constants
      auto nr_baselines = mParams.get_nr_baselines();
      auto nr_time = mParams.get_nr_timesteps();
      auto nr_polarizations = mParams.get_nr_polarizations();
      auto nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      auto subgridsize = mAlgParams.get_subgrid_size();

      // Load kernel function
      kernel::Splitter kernel_splitter(*(modules[which_module[kernel::name_splitter]]));

      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime = -omp_get_wtime();
      #endif

      // Run splitter
      for (unsigned int bl=0; bl<nr_baselines; bl+=jobsize) {
        // Prevent overflow
        jobsize = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
		
        // Number of elements in batch
        int uvw_elements     = nr_time * 3;;
        int subgrid_elements = nr_chunks * subgridsize * subgridsize * nr_polarizations;
        
        // Pointer to data for current jobs
        void *uvw_ptr     = (float *) uvw + bl * uvw_elements;
        void *subgrid_ptr = (complex<float>*) subgrids + bl * subgrid_elements;
        void *grid_ptr    = grid;

        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_splitter = -omp_get_wtime();
        #endif
        
        kernel_splitter.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);

        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_splitter += omp_get_wtime();
        total_runtime_splitter += runtime_splitter;
        #endif
        
        #if defined(REPORT_VERBOSE) 
        auxiliary::report("splitter", runtime_splitter,
                          kernel_splitter.flops(jobsize),
                          kernel_splitter.bytes(jobsize));
       #endif
        
      } // end for bl

      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime += omp_get_wtime();
      clog << endl;
      clog << "Total: splitting" << endl;
      auxiliary::report("splitter", total_runtime_splitter,
        kernel_splitter.flops(nr_baselines),
        kernel_splitter.bytes(nr_baselines));
      auxiliary::report_runtime(runtime);
      auxiliary::report_subgrids(runtime, nr_baselines);
      clog << endl;
      #endif
      
    } // run_splitter



    void CPU::run_degridder(int jobsize, void *wavenumbers, void *aterm, 
                            void *baselines, void *visibilities, void *uvw, 
                            void *spheroidal, void *subgrids)
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
      auto nr_time = mParams.get_nr_timesteps();
      auto nr_polarizations = mParams.get_nr_polarizations();
      auto nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      auto subgridsize = mAlgParams.get_subgrid_size();
      
      // Load kernel functions
      kernel::Degridder kernel_degridder(*(modules[which_module[kernel::name_degridder]]));
      kernel::GridFFT kernel_fft(*(modules[which_module[kernel::name_fft]]));
      
      // Zero visibilties: can be done when touched first time?
      #if defined(DEBUG) 
      cout << "Removed: memset(visibilities, 0, sizeof(VisibilitiesType))" << endl;
      // memset(visibilities, 0, sizeof(VisibilitiesType)); 
      #endif

      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime = -omp_get_wtime();
      #endif
      
      // Start degridder
      for (unsigned int bl=0; bl<nr_baselines; bl+=jobsize) {
        // Prevent overflow
        jobsize = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
		
        // Number of elements in batch
        int uvw_elements          = nr_time * 3;
        int visibilities_elements = nr_time * nr_channels * nr_polarizations;
        int subgrid_elements      = nr_chunks * subgridsize * subgridsize * nr_polarizations;
		
        // Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *wavenumbers_ptr  = wavenumbers;
        void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
        void *spheroidal_ptr   = spheroidal;
        void *aterm_ptr        = aterm;
        void *subgrid_ptr      = (complex<float>*) subgrids + bl * subgrid_elements;
        void *baselines_ptr    = baselines;

        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_fft = -omp_get_wtime();
        #endif
        
#if ORDER == ORDER_BL_V_U_P
        kernel_fft.run(subgridsize, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_YXP);
#elif ORDER == ORDER_BL_P_V_U
        kernel_fft.run(subgridsize, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_PYX);
#endif

        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_fft += omp_get_wtime();
        total_runtime_fft += runtime_fft;
        runtime_degridder = -omp_get_wtime();
        #endif
        
        kernel_degridder.run(jobsize, bl, subgrid_ptr, uvw_ptr, wavenumbers_ptr,
                             aterm_ptr, baselines_ptr, spheroidal_ptr, 
                             visibilities_ptr);

        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
        runtime_degridder += omp_get_wtime();
        total_runtime_degridder += runtime_degridder;
        #endif

        #if defined(REPORT_VERBOSE)
        auxiliary::report("degridder", runtime_degridder,
        kernel_degridder.flops(jobsize),
        kernel_degridder.bytes(jobsize));
        auxiliary::report("fft", runtime_fft,
        kernel_fft.flops(subgridsize, nr_baselines),
        kernel_fft.bytes(subgridsize, nr_baselines));
        #endif
        
      } // end for bl

      #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
      runtime += omp_get_wtime();
      clog << endl;
      clog << "Total: degridding" << endl;
      auxiliary::report("degridder", total_runtime_degridder,
        kernel_degridder.flops(nr_baselines),
        kernel_degridder.bytes(nr_baselines));
      auxiliary::report("fft", total_runtime_fft,
        kernel_fft.flops(subgridsize, nr_baselines),
        kernel_fft.bytes(subgridsize, nr_baselines));
      auxiliary::report_runtime(runtime);
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
      
      kernel_fft.run(gridsize, 1, grid, sign, FFT_LAYOUT_PYX);
      
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
      string parameters1 = 
        Parameters::definitions(mParams.get_nr_stations(), 
        mParams.get_nr_baselines(), 
        mParams.get_nr_timesteps(), 
        mParams.get_nr_channels(),
        mParams.get_nr_polarizations(), 
        mParams.get_field_of_view(),
        mParams.get_grid_size(), 
        mParams.get_w_planes());
      
      string parameters2 = 
        AlgorithmParameters::definitions(mAlgParams.get_subgrid_size(), 
        mAlgParams.get_chunk_size(), 
        mAlgParams.get_job_size());
      
      stringstream pp;
      pp << " -DNR_CHUNKS=" << mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      string parameters3 = pp.str();

      string parameters = " " + flags + " " + parameters1 + parameters2 
                          + parameters3;
      
      // for each shared libarary: compile the source files and put into *.so file 
      // OMP parallel?!
      for (auto libname : mInfo.get_lib_names()) {
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
