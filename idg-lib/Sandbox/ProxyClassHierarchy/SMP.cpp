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


  } // namespace proxy

} // namespace idg
