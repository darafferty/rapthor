#ifndef IDG_SMPKERNELS_H_
#define IDG_SMPKERNELS_H_

#include <cstdint>
#include <ccomplex>
#include <math.h>

#include "RuntimeWrapper.h"

namespace idg {

  //#include "KernelGridder.cpp"
  class KernelGridder {
  public:
    KernelGridder(runtime::Module &module);
    void run(int jobsize, int bl_offset, void *uvw, void *wavenumbers,
	     void *visibilities, void *spheroidal, void *aterm,
	     void *baselines, void *uvgrid);
    uint64_t flops(int jobsize);
    uint64_t bytes(int jobsize);
    
  private:
    runtime::Function _run;
    runtime::Function _flops;
    runtime::Function _bytes;
  };
  
  class KernelDegridder {
  public:
    KernelDegridder(runtime::Module &module);
    void run(int jobsize, int bl_offset, void *uvgrid, void *uvw,
	     void *wavenumbers, void *aterm, void *baselines,
	     void *spheroidal, void *visibilities);
    uint64_t flops(int jobsize);
    uint64_t bytes(int jobsize);
    
  private:
    runtime::Function _run;
    runtime::Function _flops;
    runtime::Function _bytes;
  };
  
#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)
  class KernelFFT {
  public:
    KernelFFT(runtime::Module &module);
    void run(int size, int batch, void *data, int direction, int layout);
    uint64_t flops(int size, int batch);
    uint64_t bytes(int size, int batch);
    
  private:
    runtime::Function _run;
    runtime::Function _flops;
    runtime::Function _bytes;
  };
  
  class KernelAdder {
  public:
    KernelAdder(runtime::Module &module);
    void run(int jobsize, void *uvw, void *uvgrid, void *grid);
    uint64_t flops(int jobsize);
    uint64_t bytes(int jobsize);
    
  private:
    runtime::Function _run;
    runtime::Function _flops;
    runtime::Function _bytes;
  };
  
  class KernelSplitter {
  public:
    KernelSplitter(runtime::Module &module);
    void run(int jobsize, void *uvw, void *uvgrid, void *grid);
    uint64_t flops(int jobsize);
    uint64_t bytes(int jobsize);
    
  private:
    runtime::Function _run;
    runtime::Function _flops;
    runtime::Function _bytes;
  };

} // namespace idg


#endif
