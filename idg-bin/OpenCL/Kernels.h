#include <stdint.h>
#include <complex.h>
#include <math.h>

#include <iostream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <clFFT.h>

#include "Types.h"


/*
    Gridder
*/
class KernelGridder {
    public:
        KernelGridder(cl::Program &program, const char *kernel_name);
        void launchAsync(
            cl::CommandQueue &queue, cl::Event &event, int jobsize, int bl_offset,
            cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
            cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
            cl::Buffer &d_aterm, cl::Buffer &d_baselines,
            cl::Buffer &d_subgrid);
    	static uint64_t flops(int jobsize);
		static uint64_t bytes(int jobsize);
	
	private:
	    cl::Kernel kernel;
};


/*
    FFT
*/
#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)

class KernelFFT {
	public:
        KernelFFT();
        void plan(cl::Context context, int size, int batch, int layout);
        void launchAsync(
            cl::CommandQueue &queue, cl::Event &event,
            cl::Buffer &data, clfftDirection direction);
		static uint64_t flops(int size, int batch);
		static uint64_t bytes(int size, int batch);

    private:
        bool uninitialized;
        int planned_size;
        int planned_batch;
        int planned_layout;
        clfftPlanHandle fft; 
};
