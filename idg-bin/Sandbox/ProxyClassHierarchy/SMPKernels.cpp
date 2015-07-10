#include <cstdint> // unint64_t
#include "SMPKernels.h"

// Function signatures
#define sig_degridder (void (*)(int,int,void*,void*,void*,void*,void*,void*,void*))
#define sig_gridder   (void (*)(int,int,void*,void*,void*,void*,void*,void*,void*))
#define sig_fft		  (void (*)(int,int,void*,int,int))
#define sig_adder	  (void (*)(int,void*,void*,void*))
#define sig_splitter  (void (*)(int,void*,void*,void*))
#define sig_shifter   (void (*)(int,void*))

namespace idg {

  KernelGridder::KernelGridder(runtime::Module &module) :
    _run(module,   "kernel_gridder"),
    _flops(module, "kernel_gridder_flops"),
    _bytes(module, "kernel_gridder_bytes")
  {}

  void KernelGridder::run(int jobsize, int bl_offset, void *uvw, void *wavenumbers, 
			  void *visibilities, void *spheroidal, void *aterm, 
			  void *baselines, void *uvgrid) {
    (sig_gridder (void *) _run)(jobsize, bl_offset, uvw, wavenumbers, visibilities, 
				spheroidal, aterm, baselines, uvgrid);
  }

  uint64_t KernelGridder::flops(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
  }

  uint64_t KernelGridder::bytes(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
  }


  KernelDegridder::KernelDegridder(runtime::Module &module) :
    _run(module,   "kernel_degridder"),
    _flops(module, "kernel_degridder_flops"),
    _bytes(module, "kernel_degridder_bytes")
  {}

  void KernelDegridder::run(int jobsize, int bl_offset, void *uvgrid, void *uvw, 
			    void *wavenumbers, void *aterm, void *baselines, 
			    void *spheroidal, void *visibilities) {
    (sig_degridder (void *) _run)(jobsize, bl_offset, uvgrid, uvw, wavenumbers, 
				  aterm, baselines, spheroidal, visibilities);
  }

  uint64_t KernelDegridder::flops(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
  }

  uint64_t KernelDegridder::bytes(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
  }

  KernelFFT::KernelFFT(runtime::Module &module) :
    _run(module,   "kernel_fft"),
    _flops(module, "kernel_fft_flops"),
    _bytes(module, "kernel_fft_bytes")
  {}

  void KernelFFT::run(int size, int batch, void *data, int direction, int layout) {
    (sig_fft (void *) _run)(size, batch, data, direction, layout);
  }

  uint64_t KernelFFT::flops(int size, int batch) {
    return ((uint64_t (*)(int,int)) (void *) _flops)(size, batch);
  }

  uint64_t KernelFFT::bytes(int size, int batch) {
    return ((uint64_t (*)(int,int)) (void *) _bytes)(size, batch);
  }


  KernelAdder::KernelAdder(runtime::Module &module) :
    _run(module,   "kernel_adder"),
    _flops(module, "kernel_adder_flops"),
    _bytes(module, "kernel_adder_bytes")
     {}

  void KernelAdder::run(int jobsize, void *uvw, void *uvgrid, void *grid) {
    (sig_adder (void *) _run)(jobsize, uvw, uvgrid, grid);
  }

  uint64_t KernelAdder::flops(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
  }

  uint64_t KernelAdder::bytes(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
  }

  KernelSplitter::KernelSplitter(runtime::Module &module) :
    _run(module,   "kernel_splitter"),
    _flops(module, "kernel_splitter_flops"),
    _bytes(module, "kernel_splitter_bytes")
     {}

  void KernelSplitter::run(int jobsize, void *uvw, void *uvgrid, void *grid) {
    (sig_splitter (void *) _run)(jobsize, uvw, uvgrid, grid);
  }
  
  uint64_t KernelSplitter::flops(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _flops)(jobsize);
  }

  uint64_t KernelSplitter::bytes(int jobsize) {
    return ((uint64_t (*)(int)) (void *) _bytes)(jobsize);
  }

} // namespace idg
