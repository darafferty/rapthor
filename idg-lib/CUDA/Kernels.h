#include <stdint.h>
#include <complex.h>

#include "Types.h"

#include "CU.h"
#include "CUFFT.h"


/*
    Gridder
*/
class KernelGridder {
    public:
        KernelGridder(cu::Module &module, const char *kernel);
        void launchAsync(
            cu::Stream &stream, int jobsize, int bl_offset,
            cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
            cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
            cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_baselines,
            cu::DeviceMemory &d_subgrid);
    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
	
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
            cu::DeviceMemory &d_subgrid, cu::DeviceMemory &d_uvw,
            cu::DeviceMemory &d_wavenumbers, cu::DeviceMemory &d_aterm,
            cu::DeviceMemory &d_baselines, cu::DeviceMemory &d_spheroidal,
            cu::DeviceMemory &d_visibilities);
    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
	
	private:
	    cu::Function function;
};


/*
    FFT
*/
#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)

class KernelFFT {
	public:
        KernelFFT();
        void plan(int size, int batch, int layout);
        void launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction);
		static uint64_t flops(int size, int batch);
		static uint64_t bytes(int size, int batch);

    private:
        int planned_size;
        int planned_batch;
        int planned_layout;
        cufft::C2C_2D *fft;
};


/*
    Adder
*/
class KernelAdder {
	public:
		KernelAdder(cu::Module &module, const char *kernel);
		void launchAsync(
			cu::Stream &stream, int jobsize, int bl_offset,
			cu::DeviceMemory &d_uvw,
			cu::DeviceMemory &d_subgrid,
			cu::DeviceMemory &d_grid);
		static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
		
	private:
		cu::Function function;
};


/*
    Splitter
*/
class KernelSplitter {
	public:
		KernelSplitter(cu::Module &module, const char *kernel);
		void launchAsync(
			cu::Stream &stream, int jobsize, int bl_offset,
			cu::DeviceMemory &d_uvw,
			cu::DeviceMemory &d_subgrid,
			cu::DeviceMemory &d_grid);
		static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
		
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
            cu::DeviceMemory &d_subgrid);
        static	uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
		
	private:
		cu::Function function;
};
