#include <complex>
#include "SMP.h"

using namespace std;

namespace idg {

  namespace proxy {
    
    /// Constructors
    SMP::SMP(Compiler compiler, 
	Compilerflags flags,
	Parameters params,
	ProxyInfo info) 
      : mParams(params),
	mInfo(info)
    {
      cout << "SMP::" << __func__ << endl;
      
      cout << "Compiler: " << compiler << endl;
      cout << "Compiler flags: " << flags << endl;

      // allocate subgrid data structure?

      compile(compiler, flags); 

      // load_shared_objects();
    }
    

    
    SMP::SMP(CompilerEnvironment cc, 
	     Parameters params,
	     ProxyInfo info) 
    {
      cout << "SMP::" << __func__ << endl;

      cout << cc;
      cout << params;
      cout << info;      
    } 


    SMP::~SMP() 
    {
      cout << "SMP::" << __func__ << endl;
      cerr << "~SMP() to be implemented" << endl;

      // free subgrid data structure?
      // unload modules?
      // delete .so files?
    }


    ProxyInfo SMP::default_info() 
    {
      cout << "SMP::" << __func__ << endl;

      ProxyInfo p;
      p.set_path_to_src("./src/kernels");
      p.set_path_to_lib("./lib");

      p.add_lib("Gridder.so"); // should all be with a hash to avoid conflict
      p.add_lib("Degridder.so");
      p.add_lib("FFT.so");
      p.add_lib("Adder.so");
      p.add_lib("Splitter.so");
      
      p.add_src_file_to_lib("Gridder.so", "KernelGridder.cpp");
      p.add_src_file_to_lib("Degridder.so", "KernelDegridder.cpp");
      p.add_src_file_to_lib("FFT.so", "KernelFFT.cpp");
      p.add_src_file_to_lib("Adder.so", "KernelAdder.cpp");
      p.add_src_file_to_lib("Splitter.so", "KernelSplitter.cpp");
      
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
      cout << "SMP::" << __func__ << endl;

      // allocate subgrid
      int jobsize = mParams.get_job_size();
      void* subgrids;

      grid_onto_subgrids(jobsize, visibilities, uvw, wavenumbers, aterm, 
			 spheroidal, baselines, subgrids);

      // Note: the jobsize might be different from the one above
      add_subgrids_to_grid(jobsize, uvw, subgrids, grid); 

      // free subgrid
    }


    void SMP::degrid_visibilities(void *grid,
				  void *uvw,
				  void *wavenumbers, 
				  void *aterm,
				  void *spheroidal, 
				  void *baselines,
				  void *visibilities) 
    {
      cout << "SMP::" << __func__ << endl;

      // allocate subgrids?
      int jobsize = mParams.get_job_size();
      void* subgrids;

      split_grid_into_subgrids(jobsize, uvw, subgrids, grid);

      // Note: job_size might be different
      degrid_from_subgrids(jobsize, wavenumbers, aterm, baselines, visibilities, 
			   uvw, spheroidal, subgrids); 
    }


    void SMP::transform(DomainAtoDomainB direction, void* grid) 
    {
      cout << "SMP::" << __func__ << endl;

      cout << "Direction: " << direction << endl;
      // ((void (*)(void*,int)) (void *)
      //  rw::Function(*module, FUNCTION_FFT))(grid, sign);
    }

    
    // lower-level inteface: (de)gridding split into two function calls each
    void SMP::grid_onto_subgrids(int jobsize, void *visibilities, void *uvw, 
				 void *wavenumbers, void *aterm, void *spheroidal, 
				 void *baselines, void *subgrids)
    {
      cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_gridder(jobsize, visibilities, uvw, 
		  wavenumbers, aterm, spheroidal, 
		  baselines, subgrids);
    }


    void SMP::add_subgrids_to_grid(int jobsize, void *uvw, void *subgrids, 
				   void *grid) 
    {
      cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_adder(jobsize, uvw, subgrids, grid);
    }


    void SMP::split_grid_into_subgrids(int jobsize, void *uvw, void *subgrids, 
				       void *grid)
    {
      cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_splitter(jobsize, uvw, subgrids, grid);
    }

    
    void SMP::degrid_from_subgrids(int jobsize, void *wavenumbers, void *aterm, 
				   void *baselines, void *visibilities, void *uvw, 
				   void *spheroidal, void *subgrids)
    {
      cout << "SMP::" << __func__ << endl;

      // argument checks
      // check if visibilities.size() etc... matches what is in parameters of proxy

      run_degridder(jobsize, wavenumbers, aterm, baselines, visibilities, 
		    uvw, spheroidal, subgrids);
    }


    
    void SMP::run_gridder(int jobsize, void *visibilities, void *uvw, 
			  void *wavenumbers, void *aterm, void *spheroidal, 
			  void *baselines, void *subgrids) 
    {
      cout << "SMP::" << __func__ << endl;

      // Constants
      int NR_BASELINES = mParams.get_nr_baselines();
      int NR_TIME = mParams.get_nr_timesteps();
      int NR_CHANNELS = mParams.get_nr_channels();
      int NR_POLARIZATIONS = mParams.get_nr_polarizations();
      int NR_CHUNKS = mParams.get_chunk_size();
      int SUBGRIDSIZE = mParams.get_subgrid_size();

      // Load kernel modules
      // rw::Module module_gridder(SO_GRIDDER);
      // rw::Module module_fft(SO_FFT);
      
      // Load kernel functions
      // KernelGridder kernel_gridder(module_gridder);
      // KernelFFT kernel_fft(module_fft);
      
      // Start gridder
      for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
 	// Prevent overflow
 	jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
	
 	// Number of elements in batch
 	int uvw_elements          = NR_TIME * 3;
 	int visibilities_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
 	int subgrid_elements      = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;

 	// Pointers to data for current batch
	void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
	void *wavenumbers_ptr  = wavenumbers;
 	void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
 	void *spheroidal_ptr   = spheroidal;
 	void *aterm_ptr        = aterm;
 	void *subgrids_ptr      = (complex<float>*) subgrids + bl * subgrid_elements;
	void *baselines_ptr    = baselines;
	
 	// kernel_gridder.run(jobsize, bl, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
 	// 		   spheroidal_ptr, aterm_ptr, baselines_ptr, subgrid_ptr);
	
	// #if ORDER == ORDER_BL_V_U_P
	//         kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_YXP);
	// #elif ORDER == ORDER_BL_P_V_U
	//         kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_PYX);
	// #endif
      
      } // end for bl
	
    } // run_gridder



    void SMP::run_adder(int jobsize, void *uvw, void *subgrids, void *grid) 
    {
      cout << "SMP::" << __func__ << endl;

      // Constants
      int NR_BASELINES = mParams.get_nr_baselines();
      int NR_TIME = mParams.get_nr_timesteps();
      int NR_POLARIZATIONS = mParams.get_nr_polarizations();
      int NR_CHUNKS = mParams.get_chunk_size();
      int SUBGRIDSIZE = mParams.get_subgrid_size();

      // Load kernel module
      // rw::Module module_adder(SO_ADDER);
	
      // Load kernel function
      // KernelAdder kernel_adder(module_adder);

      // Run adder
      for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
	// Prevent overflow
	jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
	// Number of elements in batch
        int uvw_elements     = NR_TIME * 3;
	int subgrid_elements = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
	// Pointer to data for current jobs
	void *uvw_ptr     = (float*) uvw + bl * uvw_elements;
	void *subgrid_ptr = (complex<float>*) subgrids + bl * subgrid_elements;
	void *grid_ptr    = grid;
	
	// kernel_adder.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);

      } // end for bl
	
    } // run_adder



    void SMP::run_splitter(int jobsize, void *uvw, void *subgrids, void *grid) 
    {
      cout << "SMP::" << __func__ << endl;

      // Constants
      int NR_BASELINES = mParams.get_nr_baselines();
      int NR_TIME = mParams.get_nr_timesteps();
      int NR_POLARIZATIONS = mParams.get_nr_polarizations();
      int NR_CHUNKS = mParams.get_chunk_size();
      int SUBGRIDSIZE = mParams.get_subgrid_size();

      // Load kernel module
      // rw::Module module_splitter(SO_SPLITTER);
      
      // Load kernel function
      // KernelSplitter kernel_splitter(module_splitter);
      
      // Run splitter
      for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
	// Prevent overflow
	jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
	// Number of elements in batch
        int uvw_elements     = NR_TIME * 3;;
	int subgrid_elements = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
	
	// Pointer to data for current jobs
	void *uvw_ptr     = (float *) uvw + bl * uvw_elements;
	void *subgrid_ptr = (complex<float>*) subgrids + bl * subgrid_elements;
	void *grid_ptr    = grid;
	
	// kernel_splitter.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);

      } // end for bl
	
    } // run_splitter



    void SMP::run_degridder(int jobsize, void *wavenumbers, void *aterm, 
			    void *baselines, void *visibilities, void *uvw, 
			    void *spheroidal, void *subgrids)
    {
      cout << "SMP::" << __func__ << endl;

      // Constants
      int NR_BASELINES = mParams.get_nr_baselines();
      int NR_CHANNELS = mParams.get_nr_channels();
      int NR_TIME = mParams.get_nr_timesteps();
      int NR_POLARIZATIONS = mParams.get_nr_polarizations();
      int NR_CHUNKS = mParams.get_chunk_size();
      int SUBGRIDSIZE = mParams.get_subgrid_size();

      // Load kernel modules
      //	rw::Module module_degridder(SO_DEGRIDDER);
      //	rw::Module module_fft(SO_FFT);
	
      // Load kernel functions
      //	KernelDegridder kernel_degridder(module_degridder);
      //	KernelFFT kernel_fft(module_fft);

      // Zero visibilties;; MP: necessary to do? Why not when access them? I.e. in the kernel
      cout << "Removed memset(visibilities, 0, sizeof(VisibilitiesType))" << endl;
      // memset(visibilities, 0, sizeof(VisibilitiesType)); 
	
      // Start degridder
      for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
	// Prevent overflow
	jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
	// Number of elements in batch
	int uvw_elements          = NR_TIME * 3;
	int visibilities_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
	int subgrid_elements      = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
	// Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *wavenumbers_ptr  = wavenumbers;
	void *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
	void *spheroidal_ptr   = spheroidal;
	void *aterm_ptr        = aterm;
	void *subgrid_ptr      = (complex<float>*) subgrids + bl * subgrid_elements;
	void *baselines_ptr    = baselines;
	
        // #if ORDER == ORDER_BL_V_U_P
	// kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_YXP);
        // #elif ORDER == ORDER_BL_P_V_U
	// kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_PYX);
        // #endif

	// kernel_degridder.run(jobsize, bl, subgrid_ptr, uvw_ptr, wavenumbers_ptr,
	// 		       aterm_ptr, baselines_ptr, spheroidal_ptr, visibilities_ptr);
	
      } // end for bl
	
    } // run_degridder



    void SMP::compile(Compiler compiler, Compilerflags flags) 
    {
      // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
      string parameters1 = ObservationParameters::definitions(mParams.get_nr_stations(), 
							      mParams.get_nr_baselines(), 
							      mParams.get_nr_timesteps(), 
							      mParams.get_nr_channels(),
							      mParams.get_nr_polarizations(), 
							      mParams.get_field_of_view()); 
      string parameters2 = AlgorithmicParameters::definitions(mParams.get_grid_size(), 
							      mParams.get_subgrid_size(), 
							      mParams.get_chunk_size(), 
							      mParams.get_job_size(), 
							      mParams.get_w_planes());
      stringstream pp;
      pp << " -DNR_CHUNKS=" << mParams.get_nr_timesteps() / mParams.get_chunk_size();
      string parameters3 = pp.str();

      string parameters = parameters1 + parameters2 + parameters3;
      
      cout << parameters << endl;

      // for each shared libarary: compile the source files and put into *.so file 
      // MP: OMP parallel?!
      for (auto libname : mInfo.get_lib_names()) {
	// create library "libname"
	cout << libname << endl;
	
	vector<string> source_files = mInfo.get_source_files(libname);
	
	if (source_files.size() != 1) {
	  cerr << "Currently only support one src file per shared object" << endl;
	}

	for (auto src : source_files) {
	  string source = mInfo.get_path_to_src() + "/" + src; 
	  cout << " " << source << " " << endl;
	  
	  string lib = mInfo.get_path_to_lib() + "/" + libname;
	  cout << " " << lib << " " << endl;

	  runtimewrapper::Source(source.c_str()).compile(compiler.c_str(), 
							 lib.c_str(), 
							 parameters.c_str());
	} // for each source file for library 

      } // for each library

    } // compile
    


  } // namespace proxy

} // namespace idg
