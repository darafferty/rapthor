#include <stdint.h>
#include <complex.h>
#include <math.h>

#include <iostream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <clFFT.h>

#include "Types.h"
#include "PerformanceCounter.h"


/*
    Gridder
*/
class KernelGridder {
    public:
        KernelGridder(cl::Program &program, const char *kernel_name, PerformanceCounter &counter);
        void launchAsync(
            cl::CommandQueue &queue, int jobsize, int bl_offset,
            cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
            cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
            cl::Buffer &d_aterm, cl::Buffer &d_baselines,
            cl::Buffer &d_subgrid);
    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
	
	private:
	    cl::Kernel kernel;
        PerformanceCounter &counter;
};


/*
    Degridder
*/
class KernelDegridder {
    public:
        KernelDegridder(cl::Program &program, const char *kernel_name, PerformanceCounter &counter);
        void launchAsync(
            cl::CommandQueue &queue, int jobsize, int bl_offset,
            cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
            cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
            cl::Buffer &d_aterm, cl::Buffer &d_baselines,
            cl::Buffer &d_subgrid);
    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
	
	private:
	    cl::Kernel kernel;
        PerformanceCounter &counter;
};


/*
    Adder
*/
class KernelAdder {
    public:
        KernelAdder(cl::Program &program, const char *kernel_name, PerformanceCounter &counter);
        void launchAsync(
            cl::CommandQueue &queue, int jobsize, int bl_offset,
            cl::Buffer &d_uvw,
            cl::Buffer &d_subgrid,
            cl::Buffer &d_grid);

    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);

	private:
	    cl::Kernel kernel;
        PerformanceCounter &counter;
};


/*
    Splitter
*/
class KernelSplitter {
    public:
        KernelSplitter(cl::Program &program, const char *kernel_name, PerformanceCounter &counter);
        void launchAsync(
            cl::CommandQueue &queue, int jobsize, int bl_offset,
            cl::Buffer &d_uvw,
            cl::Buffer &d_subgrid,
            cl::Buffer &d_grid);

    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);

	private:
	    cl::Kernel kernel;
        PerformanceCounter &counter;
};


/*
    FFT
*/
#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)

class KernelFFT {
	public:
        KernelFFT(PerformanceCounter &counter);
        void plan(cl::Context &context, int size, int batch, int layout);
        void launchAsync(
            cl::CommandQueue &queue, cl::Buffer &data, clfftDirection direction);
		static uint64_t flops(int size, int batch);
		static uint64_t bytes(int size, int batch);

    private:
        bool uninitialized;
        int planned_size;
        int planned_batch;
        int planned_layout;
        clfftPlanHandle fft;
        PerformanceCounter &counter;
};
