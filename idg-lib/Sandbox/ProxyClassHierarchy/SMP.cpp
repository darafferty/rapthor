#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime

#include "SMP.h"

using namespace std;

namespace idg {

  namespace proxy {
    
    /// Constructors
    SMP::SMP(Compiler compiler, 
	     Compilerflags flags,
	     Parameters params,
	     AlgorithmParameters algparams,
	     ProxyInfo info) 
      : mParams(params),
	mAlgParams(algparams),
	mInfo(info)
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;
      if (DEBUG) cout << "Compiler: " << compiler << endl;
      if (DEBUG) cout << "Compiler flags: " << flags << endl;
      if (DEBUG) cout << params;
      if (DEBUG) cout << algparams;

      parameter_sanity_check(); // throws exception if bad parameters

      compile(compiler, flags); 

      load_shared_objects();

      find_kernel_functions();
    }
    

    
    SMP::SMP(CompilerEnvironment cc, 
	     Parameters params,
	     AlgorithmParameters algparams,
	     ProxyInfo info) 
      : mParams(params),
	mAlgParams(algparams),
	mInfo(info)
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // find out which compiler to use
      // call SMP(compiler, flags, params, algparams)

      cerr << "Constructor not implemented yet" << endl;
    } 


    SMP::~SMP() 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

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


    AlgorithmParameters SMP::default_algparams() 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      AlgorithmParameters p;
      p.set_job_size(128);  // please set sensible value here
      p.set_subgrid_size(32); // please set sensible value here
      p.set_chunk_size(128); // please set sensible value here

      return p;
    }


    ProxyInfo SMP::default_info() 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      ProxyInfo p;

      p.set_path_to_src("./src/kernels");
      p.set_path_to_lib("./lib"); // change to use tmp dir by default

      srand(time(NULL));
      int rnd = rand();
      stringstream ss;
      ss << rnd;
      string rnd_str = ss.str();

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
    void SMP::grid_visibilities(void *visibilities, 
				void *uvw, 
				void *wavenumbers,
				void *aterm, 
				void *spheroidal, 
				void *baselines, 
				void *grid) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

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


    void SMP::degrid_visibilities(void *grid,
				  void *uvw,
				  void *wavenumbers, 
				  void *aterm,
				  void *spheroidal, 
				  void *baselines,
				  void *visibilities) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // get parameters
      unsigned int nr_baselines = mParams.get_nr_baselines();
      unsigned int nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      unsigned int subgridsize = mAlgParams.get_subgrid_size();
      unsigned int nr_polarizations = mParams.get_nr_polarizations();

      // allocate subgrids: two different versions dependingon layout?
      size_t size_subgrids = (size_t) nr_baselines*nr_chunks*subgridsize*subgridsize
	*nr_polarizations;
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


    void SMP::transform(DomainAtoDomainB direction, void* grid) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      cout << "Direction: " << direction << endl;
      // ((void (*)(void*,int)) (void *)
      //  rw::Function(*module, FUNCTION_FFT))(grid, sign);
    }

    
    // lower-level inteface: (de)gridding split into two function calls each
    void SMP::grid_onto_subgrids(int jobsize, void *visibilities, void *uvw, 
				 void *wavenumbers, void *aterm, void *spheroidal, 
				 void *baselines, void *subgrids)
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_gridder(jobsize, visibilities, uvw, 
		  wavenumbers, aterm, spheroidal, 
		  baselines, subgrids);
    }


    void SMP::add_subgrids_to_grid(int jobsize, void *uvw, void *subgrids, 
				   void *grid) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_adder(jobsize, uvw, subgrids, grid);
    }


    void SMP::split_grid_into_subgrids(int jobsize, void *uvw, void *subgrids, 
				       void *grid)
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_splitter(jobsize, uvw, subgrids, grid);
    }

    
    void SMP::degrid_from_subgrids(int jobsize, void *wavenumbers, void *aterm, 
				   void *baselines, void *visibilities, void *uvw, 
				   void *spheroidal, void *subgrids)
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_degridder(jobsize, wavenumbers, aterm, baselines, visibilities, 
		    uvw, spheroidal, subgrids);
    }


    
    void SMP::run_gridder(int jobsize, void *visibilities, void *uvw, 
			  void *wavenumbers, void *aterm, void *spheroidal, 
			  void *baselines, void *subgrids) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

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

 	kernel_gridder.run(jobsize, bl, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
 	 		   spheroidal_ptr, aterm_ptr, baselines_ptr, subgrids_ptr);
	
#if ORDER == ORDER_BL_V_U_P
	kernel_fft.run(subgridsize, jobsize, subgrids_ptr, FFTW_BACKWARD, FFT_LAYOUT_YXP);
#elif ORDER == ORDER_BL_P_V_U
	kernel_fft.run(subgridsize, jobsize, subgrids_ptr, FFTW_BACKWARD, FFT_LAYOUT_PYX);
#endif
      
      } // end for bl
	
    } // run_gridder



    void SMP::run_adder(int jobsize, void *uvw, void *subgrids, void *grid) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // Constants
      auto nr_baselines = mParams.get_nr_baselines();
      auto nr_time = mParams.get_nr_timesteps();
      auto nr_polarizations = mParams.get_nr_polarizations();
      auto nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      auto subgridsize = mAlgParams.get_subgrid_size();

      // Load kernel function
      kernel::Adder kernel_adder(*(modules[which_module[kernel::name_adder]]));

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
	
	// cout << "Calling adder" << endl;
	// cout << "jobsize: " << jobsize << endl;
	// cout << "uvw_ptr: " << uvw_ptr << endl;
	// cout << "subgrid_ptr: " << subgrid_ptr << endl;
	// cout << "grid_ptr: " << grid_ptr << endl;

	kernel_adder.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);

      } // end for bl
	
    } // run_adder



    void SMP::run_splitter(int jobsize, void *uvw, void *subgrids, void *grid) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // Constants
      auto nr_baselines = mParams.get_nr_baselines();
      auto nr_time = mParams.get_nr_timesteps();
      auto nr_polarizations = mParams.get_nr_polarizations();
      auto nr_chunks = mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      auto subgridsize = mAlgParams.get_subgrid_size();

      // Load kernel function
      kernel::Splitter kernel_splitter(*(modules[which_module[kernel::name_splitter]]));
      
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
	
	kernel_splitter.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);

      } // end for bl
	
    } // run_splitter



    void SMP::run_degridder(int jobsize, void *wavenumbers, void *aterm, 
			    void *baselines, void *visibilities, void *uvw, 
			    void *spheroidal, void *subgrids)
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

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

      // Zero visibilties: can be done when toched first time?
      if (DEBUG) cout << "Removed: memset(visibilities, 0, sizeof(VisibilitiesType))" << endl;
      // memset(visibilities, 0, sizeof(VisibilitiesType)); 
	
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
	
#if ORDER == ORDER_BL_V_U_P
	kernel_fft.run(subgridsize, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_YXP);
#elif ORDER == ORDER_BL_P_V_U
	kernel_fft.run(subgridsize, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_PYX);
#endif

	kernel_degridder.run(jobsize, bl, subgrid_ptr, uvw_ptr, wavenumbers_ptr,
			     aterm_ptr, baselines_ptr, spheroidal_ptr, visibilities_ptr);
	
      } // end for bl
	
    } // run_degridder


    void SMP::compile(Compiler compiler, Compilerflags flags) 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
      string parameters1 = Parameters::definitions(
                           mParams.get_nr_stations(), 
			   mParams.get_nr_baselines(), 
			   mParams.get_nr_timesteps(), 
			   mParams.get_nr_channels(),
			   mParams.get_nr_polarizations(), 
			   mParams.get_field_of_view(),
                           mParams.get_grid_size(), 
			   mParams.get_w_planes());

      string parameters2 = AlgorithmParameters::definitions(
			   mAlgParams.get_subgrid_size(), 
			   mAlgParams.get_chunk_size(), 
			   mAlgParams.get_job_size());

      stringstream pp;
      pp << " -DNR_CHUNKS=" << mParams.get_nr_timesteps() / mAlgParams.get_chunk_size();
      string parameters3 = pp.str();

      string parameters = " " + flags + " " + parameters1 + parameters2 + parameters3;
      
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


    
    void SMP::parameter_sanity_check() 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      // assert: subgrid_size <= grid_size
      // assert: job_size <= ?
      // [...]
    }

    
    void SMP::load_shared_objects() 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;

      for (auto libname : mInfo.get_lib_names()) {
	string lib = mInfo.get_path_to_lib() + "/" + libname;

	if (DEBUG) cout << "Loading: " << libname << endl;

	modules.push_back(new runtime::Module(lib.c_str()));
      }
    }


    /// maps name -> index in modules that contain that symbol 
    void SMP::find_kernel_functions() 
    {
      if (DEBUG) cout << "SMP::" << __func__ << endl;
      
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
