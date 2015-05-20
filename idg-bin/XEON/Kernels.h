#include <stdint.h>
#include <complex.h>
#include <math.h>

#include "RW.h"

//#include "KernelGridder.cpp"
class KernelGridder {
    public:
        KernelGridder(rw::Module &module);
        void run(
            int jobsize, int bl_offset, void *uvw, void *wavenumbers,
            void *visibilities, void *spheroidal, void *aterm,
            void *baselines, void *uvgrid);
    	uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
	
	private:
        rw::Function _run;
        rw::Function _flops;
        rw::Function _bytes;
};

class KernelDegridder {
    public:
        KernelDegridder(rw::Module &module);
        void run(
            int jobsize, int bl_offset, void *uvgrid, void *uvw,
            void *wavenumbers, void *aterm, void *baselines,
            void *spheroidal, void *visibilities);
    	uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
	
	private:
        rw::Function _run;
        rw::Function _flops;
        rw::Function _bytes;
};

#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)
class KernelFFT {
	public:
		KernelFFT(rw::Module &module);
		void run(int size, int batch, void *data, int direction, int layout);
		uint64_t flops(int size, int batch);
		uint64_t bytes(int size, int batch);
		
	private:
        rw::Function _run;
        rw::Function _flops;
        rw::Function _bytes;
};

class KernelAdder {
	public:
		KernelAdder(rw::Module &module);
		void run(int jobsize, void *uvw, void *uvgrid, void *grid);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
        rw::Function _run;
        rw::Function _flops;
        rw::Function _bytes;
};

class KernelSplitter {
	public:
		KernelSplitter(rw::Module &module);
		void run(int jobsize, void *uvw, void *uvgrid, void *grid);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
        rw::Function _run;
        rw::Function _flops;
        rw::Function _bytes;
};
