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


      // ((void (*)(int,void*,void*,void*,void*,void*,void*,void*)) (void *)
      //  rw::Function(*module, FUNCTION_GRIDDER))(jobsize, visibilities, 
      // 						uvw, wavenumbers, aterm, 
      // 						spheroidal, baselines, 
      // 						subgrid);
      
      // ((void (*)(int,void*,void*,void*)) (void *)
      //  rw::Function(*module, FUNCTION_ADDER))(jobsize, coordinates, subgrid, grid);      
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

      // ((void (*)(int,void*,void*,void*)) (void *)
      //  rw::Function(*module, FUNCTION_SPLITER))(jobsize, coordinates, subgrid, grid);

      // ((void (*)(int,void*,void*,void*,void*,void*,void*,void*)) (void *)
      //  rw::Function(*module, FUNCTION_DEGRIDDER))(jobsize, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, subgrid);
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


  } // namespace proxy

} // namespace idg
