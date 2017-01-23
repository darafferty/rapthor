#include "Kernels.h"

#define ENABLE_PERFORMANCE_COUNTERS 1

using namespace std;

namespace idg {
    namespace kernel {
        namespace opencl {

            // Gridder class
            Gridder::Gridder(
                cl::Program &program,
                const cl::NDRange &local_size) :
                kernel(program, name_gridder.c_str()),
                local_size(local_size) {}

            void Gridder::launchAsync(
                cl::CommandQueue &queue,
                int nr_timesteps,
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
                PerformanceCounter &counter) {
                int local_size_x = local_size[0];
                int local_size_y = local_size[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel.setArg(0,  gridsize);
                kernel.setArg(1,  imagesize);
                kernel.setArg(2,  w_offset);
                kernel.setArg(3,  nr_channels);
                kernel.setArg(4,  nr_stations);
                kernel.setArg(5,  d_uvw);
                kernel.setArg(6,  d_wavenumbers);
                kernel.setArg(7,  d_visibilities);
                kernel.setArg(8,  d_spheroidal);
                kernel.setArg(9,  d_aterm);
                kernel.setArg(10, d_metadata);
                kernel.setArg(11, d_subgrid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, &event);
                    #if ENABLE_PERFORMANCE_COUNTERS
                    //counter.doOperation(event, "gridder", flops(nr_timesteps, nr_subgrids), bytes(nr_timesteps, nr_subgrids));
                    #endif
                } catch (cl::Error &error) {
                    std::cerr << "Error launching gridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }


            // Degridder class
            Degridder::Degridder(
                cl::Program &program,
                const cl::NDRange &local_size) :
                kernel(program, name_degridder.c_str()),
                local_size(local_size) {}

            void Degridder::launchAsync(
                cl::CommandQueue &queue,
                int nr_timesteps,
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
                PerformanceCounter &counter) {
                int local_size_x = local_size[0];
                int local_size_y = local_size[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel.setArg(0,  gridsize);
                kernel.setArg(1,  imagesize);
                kernel.setArg(2,  w_offset);
                kernel.setArg(3,  nr_channels);
                kernel.setArg(4,  nr_stations);
                kernel.setArg(5,  d_uvw);
                kernel.setArg(6,  d_wavenumbers);
                kernel.setArg(7,  d_visibilities);
                kernel.setArg(8,  d_spheroidal);
                kernel.setArg(9,  d_aterm);
                kernel.setArg(10, d_metadata);
                kernel.setArg(11, d_subgrid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, &event);
                    #if ENABLE_PERFORMANCE_COUNTERS
                    //counter.doOperation(event, "degridder", flops(nr_timesteps, nr_subgrids), bytes(nr_timesteps, nr_subgrids));
                    #endif
                } catch (cl::Error &error) {
                    std::cerr << "Error launching degridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }


            // GridFFT class
            GridFFT::GridFFT(
                unsigned int nr_correlations) :
                nr_correlations(nr_correlations)
            {
                uninitialized = true;
            }

            GridFFT::~GridFFT() {
                clfftDestroyPlan(&fft);
            }

            void GridFFT::plan(
                cl::Context &context,
                cl::CommandQueue &queue,
                int size, int batch)
            {
                // Check wheter a new plan has to be created
                if (uninitialized ||
                    size  != planned_size ||
                    batch != planned_batch) {
                    // Destroy old plan (if any)
                    if (!uninitialized) {
                        clfftDestroyPlan(&fft);
                    }

                    // Create new plan
                    size_t lengths[2] = {(size_t) size, (size_t) size};
                    clfftCreateDefaultPlan(&fft, context(), CLFFT_2D, lengths);

                    // Set plan parameters
                    clfftSetPlanPrecision(fft, CLFFT_SINGLE);
                    clfftSetLayout(fft, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
                    clfftSetResultLocation(fft, CLFFT_INPLACE);
                    int distance = size*size;
                    clfftSetPlanDistance(fft, distance, distance);
                    clfftSetPlanBatchSize(fft, batch * nr_correlations);

                    // Update parameters
                    planned_size = size;
                    planned_batch = batch;

                    // Bake plan
                    clfftStatus status = clfftBakePlan(fft, 1, &queue(), NULL, NULL);
                    if (status != CL_SUCCESS) {
                        std::cerr << "Error baking fft plan" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                uninitialized = false;
            }

            void GridFFT::launchAsync(
                cl::CommandQueue &queue,
                cl::Buffer &d_data,
                clfftDirection direction)
            {
                clfftStatus status = clfftEnqueueTransform(fft, direction, 1, &queue(), 0, NULL, NULL, &d_data(), NULL, NULL);
                if (status != CL_SUCCESS) {
                    std::cerr << "Error enqueing fft plan" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void GridFFT::launchAsync(
                cl::CommandQueue &queue,
                cl::Buffer &d_data,
                clfftDirection direction,
                PerformanceCounter &counter,
                const char *name)
            {
                queue.enqueueMarkerWithWaitList(NULL, &start);
                clfftStatus status = clfftEnqueueTransform(fft, direction, 1, &queue(), 0, NULL, NULL, &d_data(), NULL, NULL);
                queue.enqueueMarkerWithWaitList(NULL, &end);
                //counter.doOperation(start, end, name, flops(planned_size, planned_batch), bytes(planned_size, planned_batch));
                if (status != CL_SUCCESS) {
                    std::cerr << "Error enqueing fft plan" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }


            // Adder class
            Adder::Adder(
                cl::Program &program,
                const cl::NDRange &local_size) :
                kernel(program, name_adder.c_str()),
                local_size(local_size) {}

            void Adder::launchAsync(
                cl::CommandQueue &queue,
                int nr_subgrids,
                int gridsize,
                cl::Buffer d_metadata,
                cl::Buffer d_subgrid,
                cl::Buffer d_grid,
                PerformanceCounter &counter) {
                int local_size_x = local_size[0];
                int local_size_y = local_size[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel.setArg(0, gridsize);
                kernel.setArg(1, d_metadata);
                kernel.setArg(2, d_subgrid);
                kernel.setArg(3, d_grid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, &event);
                    #if ENABLE_PERFORMANCE_COUNTERS
                    //counter.doOperation(event, "adder", flops(nr_subgrids), bytes(nr_subgrids));
                    #endif
                } catch (cl::Error &error) {
                    std::cerr << "Error launching adder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }


            // Splitter class
            Splitter::Splitter(
                cl::Program &program,
                const cl::NDRange &local_size):
                kernel(program, name_splitter.c_str()),
                local_size(local_size) {}

            void Splitter::launchAsync(
                cl::CommandQueue &queue,
                int nr_subgrids,
                int gridsize,
                cl::Buffer d_metadata,
                cl::Buffer d_subgrid,
                cl::Buffer d_grid,
                PerformanceCounter &counter) {
                int local_size_x = local_size[0];
                int local_size_y = local_size[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel.setArg(0, gridsize);
                kernel.setArg(1, d_metadata);
                kernel.setArg(2, d_subgrid);
                kernel.setArg(3, d_grid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, &event);
                    #if ENABLE_PERFORMANCE_COUNTERS
                    //counter.doOperation(event, "splitter", flops(nr_subgrids), bytes(nr_subgrids));
                    #endif
                } catch (cl::Error &error) {
                    std::cerr << "Error launching splitter: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }


            // Scaler class
            Scaler::Scaler(
                cl::Program &program,
                const cl::NDRange &local_size) :
                kernel(program, name_scaler.c_str()),
                local_size(local_size) {}

            void Scaler::launchAsync(
                cl::CommandQueue &queue,
                int nr_subgrids,
                cl::Buffer d_subgrid,
                PerformanceCounter &counter) {
                int local_size_x = local_size[0];
                int local_size_y = local_size[1];
                cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
                kernel.setArg(0, d_subgrid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, &event);
                    #if ENABLE_PERFORMANCE_COUNTERS
                    //counter.doOperation(event, "scaler", flops(nr_subgrids), bytes(nr_subgrids));
                    #endif
                } catch (cl::Error &error) {
                    std::cerr << "Error launching scaler: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

        } // namespace opencl
    } // namespace kernel
} // namespace idg
