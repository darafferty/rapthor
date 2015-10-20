#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <clFFT.h>

#include "Parameters.h"
#include "PerformanceCounter.h"

namespace idg {

    namespace kernel {

        // define the kernel function names
        static const std::string name_gridder   = "kernel_gridder";
        static const std::string name_degridder = "kernel_degridder";
        static const std::string name_adder     = "kernel_adder";
        static const std::string name_splitter  = "kernel_splitter";

        class Gridder {
            public:
                Gridder(cl::Program &program, Parameters &parameters);
                void launchAsync(
                    cl::CommandQueue &queue, int jobsize, float w_offset,
                    cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
                    cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
                    cl::Buffer &d_aterm, cl::Buffer &d_metadata,
                    cl::Buffer &d_subgrid,
                    PerformanceCounter &counter);
            	uint64_t flops(int jobsize);
        		uint64_t bytes(int jobsize);

        	private:
        	    cl::Kernel kernel;
                Parameters &parameters;
        };


        class Degridder {
            public:
                Degridder(cl::Program &program, Parameters &parameters);
                void launchAsync(
                    cl::CommandQueue &queue, int jobsize, float w_offset,
                    cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
                    cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
                    cl::Buffer &d_aterm, cl::Buffer &d_metadata,
                    cl::Buffer &d_subgrid,
                    PerformanceCounter &counter);
               	uint64_t flops(int jobsize);
        		uint64_t bytes(int jobsize);

        	private:
                cl::Kernel kernel;
                Parameters &parameters;
        };


        class GridFFT {
        	public:
                GridFFT(Parameters &parameters);
                void plan(
                    cl::Context &context, int size, int batch);
                void launchAsync(
                    cl::CommandQueue &queue, cl::Buffer &d_data, clfftDirection direction);
        		uint64_t flops(int size, int batch);
        		uint64_t bytes(int size, int batch);
                double runtime();

            private:
                bool uninitialized;
                Parameters &parameters;
                int planned_size;
                int planned_batch;
                clfftPlanHandle fft;
                cl::Event event_start;
                cl::Event event_end;
        };
    } // namespace kernel
} // namespace idg

#endif
