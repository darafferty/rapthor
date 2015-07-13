#include <cstdint> // unint64_t
#include "SMPKernels.h"

namespace idg {

  namespace kernel {

    // Gridder class
    Gridder::Gridder(runtime::Module &module) :
      _run(module,   name_gridder.c_str()),
      _flops(module, name_gridder_flops.c_str()),
      _bytes(module, name_gridder_bytes.c_str())
    {}

    void Gridder::run(int jobsize, int bl_offset, void *uvw, void *wavenumbers, 
		      void *visibilities, void *spheroidal, void *aterm, 
		      void *baselines, void *uvgrid) {
      (sig_gridder (void *) _run)(jobsize, bl_offset, uvw, wavenumbers, visibilities, 
				  spheroidal, aterm, baselines, uvgrid);
    }

    uint64_t Gridder::flops(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
    }

    uint64_t Gridder::bytes(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
    }


    // Degridder class
    Degridder::Degridder(runtime::Module &module) :
      _run(module,   name_degridder.c_str()),
      _flops(module, name_degridder_flops.c_str()),
      _bytes(module, name_degridder_bytes.c_str())
    {}

    void Degridder::run(int jobsize, int bl_offset, void *uvgrid, void *uvw, 
			void *wavenumbers, void *aterm, void *baselines, 
			void *spheroidal, void *visibilities) {
      (sig_degridder (void *) _run)(jobsize, bl_offset, uvgrid, uvw, wavenumbers, 
				    aterm, baselines, spheroidal, visibilities);
    }

    uint64_t Degridder::flops(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
    }

    uint64_t Degridder::bytes(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
    }

    
    // GridFFT class
    GridFFT::GridFFT(runtime::Module &module) :
      _run(module,   name_fft.c_str()),
      _flops(module, name_fft_flops.c_str()),
      _bytes(module, name_fft_bytes.c_str())
    {}

    void GridFFT::run(int size, int batch, void *data, int direction, int layout) {
      (sig_fft (void *) _run)(size, batch, data, direction, layout);
    }

    uint64_t GridFFT::flops(int size, int batch) {
      return ((uint64_t (*)(int,int)) (void *) _flops)(size, batch);
    }

    uint64_t GridFFT::bytes(int size, int batch) {
      return ((uint64_t (*)(int,int)) (void *) _bytes)(size, batch);
    }


    // Adder class
    Adder::Adder(runtime::Module &module) :
      _run(module,   name_adder.c_str()),
      _flops(module, name_adder_flops.c_str()),
      _bytes(module, name_adder_bytes.c_str())
    {}

    void Adder::run(int jobsize, void *uvw, void *uvgrid, void *grid) {
      (sig_adder (void *) _run)(jobsize, uvw, uvgrid, grid);
    }

    uint64_t Adder::flops(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
    }

    uint64_t Adder::bytes(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
    }

    
    // Splitter class
    Splitter::Splitter(runtime::Module &module) :
      _run(module,   name_splitter.c_str()),
      _flops(module, name_splitter_flops.c_str()),
      _bytes(module, name_splitter_bytes.c_str())
    {}

    void Splitter::run(int jobsize, void *uvw, void *uvgrid, void *grid) {
      (sig_splitter (void *) _run)(jobsize, uvw, uvgrid, grid);
    }
  
    uint64_t Splitter::flops(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
    }

    uint64_t Splitter::bytes(int jobsize) {
      return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
    }

  } // namespace kernel

} // namespace idg
