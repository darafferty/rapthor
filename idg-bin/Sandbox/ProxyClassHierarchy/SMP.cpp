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

      compile(compiler, flags); 
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
    }


    ProxyInfo SMP::default_info() 
    {
      cout << "SMP::" << __func__ << endl;

      ProxyInfo p;
      p.set_path_to_src("./src/kernels");
      p.set_path_to_lib("./lib");

      p.add_lib("Gridder.so"); // should all be with a hash
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

      // add_subgrids_to_grid(); // adder

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

    }


    void SMP::transform(DomainAtoDomainB direction, void* grid) 
    {
      cout << "SMP::" << __func__ << endl;

      cout << "Direction: " << direction << endl;
      // ((void (*)(void*,int)) (void *)
      //  rw::Function(*module, FUNCTION_FFT))(grid, sign);
    }


    // gridder
    // adder
    // splitter
    // degridder



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

      // for each shared libarary *.s: compile the source files and put into *.so file 
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

	cout << "uvw elements = " << uvw_elements << endl;
	cout << "visibilities_elements = " << visibilities_elements << endl;
	cout << "subgrid_elements = " << subgrid_elements << endl;
	
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
    


  } // namespace proxy

} // namespace idg
