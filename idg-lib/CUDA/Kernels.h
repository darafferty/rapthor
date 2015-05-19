#include <stdint.h>
#include <complex.h>

#include "Types.h"

#include "CU.h"
#include "CUFFT.h"

#define GRID_DISTRIBUTION   32

/*
    Gridder
*/
class KernelGridder {
    public:
        KernelGridder(cu::Module &module, const char *kernel);
        void launchAsync(
            cu::Stream &stream, int jobsize, int bl_offset,
            cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_offset,
            cu::DeviceMemory &d_wavenumbers, cu::DeviceMemory &d_visibilities,
            cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_aterm,
            cu::DeviceMemory &d_baselines, cu::DeviceMemory &d_uvgrid);
    	uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
	
	private:
	    cu::Function function;
};


/*
    Degridder
*/
class KernelDegridder {
    public:
        KernelDegridder(cu::Module &module, const char *kernel);
        void launchAsync(
            cu::Stream &stream, int jobsize, int bl_offset,
            cu::DeviceMemory &d_uvgrid, cu::DeviceMemory &d_uvw,
            cu::DeviceMemory &d_offset, cu::DeviceMemory &d_wavenumbers,
            cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_baselines,
            cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_visibilities);
    	uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
	
	private:
	    cu::Function function;
};


/*
    FFT
*/
class KernelFFT {
	public:
		void launchAsync(cu::Stream &stream, int jobsize,
			cu::DeviceMemory &d_uvgrid, int direction);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
};


/*
    Adder
*/
class KernelAdder {
	public:
		KernelAdder(cu::Module &module, const char *kernel);
		void launchAsync(
			cu::Stream &stream, int jobsize, int bl_offset,
			cu::DeviceMemory &d_coordinates,
			cu::DeviceMemory &d_uvgrid,
			cu::DeviceMemory &d_grid);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
		cu::Function function;
};


/*
    Shifter
*/
class KernelShifter {
    public:
        KernelShifter(cu::Module &module, const char *kernel);
        void launchAsync(
            cu::Stream &stream, int jobsize,
            cu::DeviceMemory &d_uvgrid);
            		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
		cu::Function function;
};
