#ifndef IDG_OPENCL_KERNELS_H_
#define IDG_OPENCL_KERNELS_H_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include <clFFT.h>

#include "idg-common.h"

#include "PerformanceCounter.h"

namespace idg {
    namespace kernel {
        namespace opencl {

			// define the kernel function names
			static const std::string name_gridder   = "kernel_gridder";
			static const std::string name_degridder = "kernel_degridder";
			static const std::string name_adder     = "kernel_adder";
			static const std::string name_splitter  = "kernel_splitter";
			static const std::string name_scaler    = "kernel_scaler";

            class Gridder {
                public:
                    Gridder(
						cl::Program &program,
						const cl::NDRange &local_size);
                     void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_baselines,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cl::Buffer &d_uvw,
                        cl::Buffer &d_wavenumbers,
                        cl::Buffer &d_visibilities,
                        cl::Buffer &d_spheroidal,
                        cl::Buffer &d_aterm,
                        cl::Buffer &d_metadata,
                        cl::Buffer &d_subgrid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };


            class Degridder {
                public:
                    Degridder(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_baselines,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cl::Buffer &d_uvw,
                        cl::Buffer &d_wavenumbers,
                        cl::Buffer &d_visibilities,
                        cl::Buffer &d_spheroidal,
                        cl::Buffer &d_aterm,
                        cl::Buffer &d_metadata,
                        cl::Buffer &d_subgrid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };


            class GridFFT {
                public:
                    GridFFT(
                        unsigned int nr_correlations);
                    ~GridFFT();
                    void plan(
                        cl::Context &context, cl::CommandQueue &queue,
                        int size, int batch);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        cl::Buffer &d_data,
                        clfftDirection direction);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        cl::Buffer &d_data,
                        clfftDirection direction,
                        PerformanceCounter &counter,
                        const char *name);
                    void shift(std::complex<float> *data);
                    void scale(std::complex<float> *data, std::complex<float> scale);

                private:
                    bool uninitialized;
                    unsigned int nr_correlations;
                    int planned_size;
                    int planned_batch;
                    clfftPlanHandle fft;
                    cl::Event start;
                    cl::Event end;
            };

            class Adder {
                public:
                    Adder(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_subgrids,
                        int gridsize,
                        cl::Buffer d_metadata,
                        cl::Buffer d_subgrid,
                        cl::Buffer d_grid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };

            class Splitter {
                public:
                    Splitter(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_subgrids,
                        int gridsize,
                        cl::Buffer d_metadata,
                        cl::Buffer d_subgrid,
                        cl::Buffer d_grid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };

            class Scaler {
                public:
                    Scaler(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_subgrids,
                        cl::Buffer d_subgrid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };

        } // namespace opencl
    } // namespace kernel
} // namespace idg

#endif
