#include "SMP.h"

using namespace std;

namespace idg {

  namespace proxy {
    
    /// Constructors
    SMP::SMP(Compiler compiler, 
	Compilerflags flags,
	CompileTimeConstants constants,
	ProxyInfo info) 
    {
      cout << "SMP::" << __func__ << endl;
      
      cout << "Compiler: " << compiler << endl;
      cout << "Compiler flags: " << flags << endl;
      cout << constants;
      cout << info;

      compile(compiler, flags, constants, info); 
    }
    

    
    SMP::SMP(CompilerEnvironment cc, 
	     CompileTimeConstants constants,
	     ProxyInfo info) 
    {
      cout << "SMP::" << __func__ << endl;

      cout << cc;
      cout << constants;
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
      p.set_path_to_src("./src");
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
      int jobsize = 100;
      void* subgrid;

      // grid_onto_subgrid(jobsize, visibilities, uvw, wavenumbers, aterm, 
      // 			spheroidal, baselines, subgrid);

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



    void SMP::compile(Compiler compiler, Compilerflags flags, 
		      CompileTimeConstants constants, ProxyInfo info) 
    {
      // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
      string parameters1 = ObservationParameters::definitions(constants.get_nr_stations(), 
							      constants.get_nr_baselines(), 
							      constants.get_nr_timesteps(), 
							      constants.get_nr_channels(),
							      constants.get_nr_polarizations(), 
							      constants.get_field_of_view()); 
      string parameters2 = AlgorithmicParameters::definitions(constants.get_grid_size(), 
							      constants.get_subgrid_size(), 
							      constants.get_chunk_size(), 
							      constants.get_job_size(), 
							      constants.get_w_planes());
      stringstream pp;
      pp << " -DNR_CHUNKS=" << constants.get_nr_timesteps() / constants.get_chunk_size();
      string parameters3 = pp.str();

      string parameters = parameters1 + parameters2 + parameters3;
      
      cout << parameters << endl;

      /// below needs to be modified

      // Compile wrapper
      string options = parameters + " " + flags + " " + "-I.. -I../Common" + " " +
	"RuntimeWrapper.cpp"     + " " + "src/Kernels.cpp";
      //      runtimewrapper::Source("src/Wrapper.cpp").compile(compiler.c_str(), "lib/Wrapper.so", options.c_str());
      
      // Load module
      // module = new runtimewrapper::Module("./lib/Wrapper.so");  // have to be relased in destructor
      
      // Initialize module
      // ((void (*)(const char*, const char *)) (void *) runtimewrapper::Function(*module, "init"))(compiler.c_str(), flags.c_str());
    }



    void SMP::grid_onto_subgrid(int jobsize, void *visibilities, void *uvw, 
				void *wavenumbers, void *aterm, void *spheroidal, 
				void *baselines, void *subgrid)
    {
      run_gridder(jobsize, visibilities, uvw, 
		  wavenumbers, aterm, spheroidal, 
		  baselines, subgrid);
    }

    
    void SMP::run_gridder(int jobsize, void *visibilities, void *uvw, 
			  void *wavenumbers, void *aterm, void *spheroidal, 
			  void *baselines, void *subgrid) 
    {
//       // HACK:
//       // int NR_BASELINES =

//       // Load kernel modules
//       // rw::Module module_gridder(SO_GRIDDER);
//       // rw::Module module_fft(SO_FFT);
      
//       // Load kernel functions
//       // KernelGridder kernel_gridder(module_gridder);
//       // KernelFFT kernel_fft(module_fft);
      
//       // Start gridder
//       for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
// 	// Prevent overflow
// 	jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
	
// 	// Number of elements in batch
// 	int uvw_elements          = NR_TIME * 3;
// 	int visibilities_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
// 	int subgrid_elements      = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
	
// 	// Pointers to data for current batch
//         void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
//         void *wavenumbers_ptr  = wavenumbers;
// 	void *visibilities_ptr = (float complex *) visibilities + bl * visibilities_elements;
// 	void *spheroidal_ptr   = spheroidal;
// 	void *aterm_ptr        = aterm;
// 	void *subgrid_ptr      = (float complex *) subgrid + bl * subgrid_elements;
// 	void *baselines_ptr    = baselines;
	
// 	// kernel_gridder.run(jobsize, bl, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
// 	// 		   spheroidal_ptr, aterm_ptr, baselines_ptr, subgrid_ptr);
	
// // #if ORDER == ORDER_BL_V_U_P
// //         kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_YXP);
// // #elif ORDER == ORDER_BL_P_V_U
// //         kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_PYX);
// // #endif
//       } 
	
    } // run_gridder
    


  } // namespace proxy

} // namespace idg
