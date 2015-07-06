
#include "SMP.h"

namespace idg {

  namespace proxy {
    
    void SMP::grid_visibilities(void *visibilities, 
				void *uvw, 
				void *wavenumbers,
				void *aterm, 
				void *spheroidal, 
				void *baselines, 
				void *grid) 
    {
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
      // ((void (*)(int,void*,void*,void*)) (void *)
      //  rw::Function(*module, FUNCTION_SPLITER))(jobsize, coordinates, subgrid, grid);

      // ((void (*)(int,void*,void*,void*,void*,void*,void*,void*)) (void *)
      //  rw::Function(*module, FUNCTION_DEGRIDDER))(jobsize, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, subgrid);
    }


    void SMP::transform(DomainAtoDomainB direction, void* grid) 
    {
      // ((void (*)(void*,int)) (void *)
      //  rw::Function(*module, FUNCTION_FFT))(grid, sign);
    }


  } // namespace proxy

} // namespace idg
