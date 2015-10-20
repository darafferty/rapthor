#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

using namespace std;

namespace idg {

    namespace kernel {

        double compute_runtime(cl::Event &event_start, cl::Event &event_end) {
            double runtime = 0;
            cl_ulong start, end;
            if (clGetEventProfilingInfo(event_start(), CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(event_end(), CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
                runtime = (end - start) * 1e-9;
            }
            return runtime;
        }

        double compute_runtime(cl::Event &event) {
            return compute_runtime(event, event);
        }

        // Gridder class
        Gridder::Gridder(cl::Program &program, Parameters &parameters) :
            kernel(program, name_gridder.c_str()),
            parameters(parameters) {}

        void Gridder::launchAsync(
            cl::CommandQueue &queue, int jobsize, float w_offset,
            cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
            cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
            cl::Buffer &d_aterm, cl::Buffer &d_metadata,
            cl::Buffer &d_subgrid) {
            cl::NDRange globalSize(32 * jobsize, 4);
            cl::NDRange localSize(32, 4);
            kernel.setArg(0, w_offset);
            kernel.setArg(1, d_uvw);
            kernel.setArg(2, d_wavenumbers);
            kernel.setArg(3, d_visibilities);
            kernel.setArg(4, d_spheroidal);
            kernel.setArg(5, d_aterm);
            kernel.setArg(6, d_metadata);
            kernel.setArg(7, d_subgrid);
            try {
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
            } catch (cl::Error &error) {
                std::cerr << "Error launching gridder: " << error.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        uint64_t Gridder::flops(int jobsize) {
            int subgridsize = parameters.get_subgrid_size();
            int nr_timesteps = parameters.get_nr_timesteps();
            int nr_channels = parameters.get_nr_channels();
            int nr_polarizations = parameters.get_nr_polarizations();
            uint64_t flops = 0;
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase index
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase offset
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * 2; // phase
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
            flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
            flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
            flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
            return flops;
        }

        uint64_t Gridder::bytes(int jobsize) {
            int subgridsize = parameters.get_subgrid_size();
            int nr_timesteps = parameters.get_nr_timesteps();
            int nr_channels = parameters.get_nr_channels();
            int nr_polarizations = parameters.get_nr_polarizations();
            uint64_t bytes = 0;
            bytes += 1ULL * jobsize * nr_timesteps * 3 * sizeof(float); // uvw
            bytes += 1ULL * jobsize * nr_timesteps * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities
            bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * 2 * sizeof(float); // subgrids
            return bytes;
        }

        double Gridder::runtime() {
            return compute_runtime(event);
        }


        // Degridder class
        Degridder::Degridder(cl::Program &program, Parameters &parameters) :
            kernel(program, name_degridder.c_str()),
            parameters(parameters) {}

        void Degridder::launchAsync(
            cl::CommandQueue &queue, int jobsize, float w_offset,
            cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
            cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
            cl::Buffer &d_aterm, cl::Buffer &d_metadata,
            cl::Buffer &d_subgrid) {
            // IF wgSize IS MODIFIED, ALSO MODIFY NR_THREADS in KernelDegridder.cl
            int wgSize = 256;
            cl::NDRange globalSize(jobsize * wgSize);
            cl::NDRange localSize(wgSize);
            kernel.setArg(0, w_offset);
            kernel.setArg(1, d_uvw);
            kernel.setArg(2, d_wavenumbers);
            kernel.setArg(3, d_visibilities);
            kernel.setArg(4, d_spheroidal);
            kernel.setArg(5, d_aterm);
            kernel.setArg(6, d_metadata);
            kernel.setArg(7, d_subgrid);
            try {
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
            } catch (cl::Error &error) {
                std::cerr << "Error launching degridder: " << error.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        uint64_t Degridder::flops(int jobsize) {
            int subgridsize = parameters.get_subgrid_size();
            int nr_timesteps = parameters.get_nr_timesteps();
            int nr_channels = parameters.get_nr_channels();
            int nr_polarizations = parameters.get_nr_polarizations();
            uint64_t flops = 0;
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase index
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase offset
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * 2; // phase
            flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
            flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
            flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
            flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
            return flops;
        }

        uint64_t Degridder::bytes(int jobsize) {
            int subgridsize = parameters.get_subgrid_size();
            int nr_timesteps = parameters.get_nr_timesteps();
            int nr_channels = parameters.get_nr_channels();
            int nr_polarizations = parameters.get_nr_polarizations();
            uint64_t bytes = 0;
            bytes += 1ULL * jobsize * nr_timesteps * 3 * sizeof(float); // uvw
            bytes += 1ULL * jobsize * nr_timesteps * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities
            bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * 2 * sizeof(float); // subgrids
            return bytes;
        }

        double Degridder::runtime() {
            return compute_runtime(event);
        }


        // GridFFT class
        GridFFT::GridFFT(Parameters &parameters) : parameters(parameters) {
            uninitialized = true;
        }

        void GridFFT::plan(cl::Context &context, int size, int batch) {
            // Check wheter a new plan has to be created
            if (uninitialized ||
               size  != planned_size ||
               batch != planned_batch) {
                // Create new plan
                size_t lengths[2] = {(size_t) size, (size_t) size};
                clfftCreateDefaultPlan(&fft, context(), CLFFT_2D, lengths);
                clfftSetPlanBatchSize(fft, batch);
                size_t dist = size * size;
                clfftSetPlanDistance(fft, dist, dist);

                // Update parameters
                planned_size = size;
                planned_batch = batch;
            }
        }

        void GridFFT::launchAsync(
            cl::CommandQueue &queue, cl::Buffer &d_data, clfftDirection direction) {
            queue.enqueueMarkerWithWaitList(NULL, &event_start);
            clfftEnqueueTransform(fft, direction, 1, &queue(), 0, NULL, NULL, &d_data(), NULL, NULL);
            queue.enqueueMarkerWithWaitList(NULL, &event_end);
        }

        uint64_t GridFFT::flops(int size, int batch) {
            int nr_polarizations = parameters.get_nr_polarizations();
        	return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
        }

        uint64_t GridFFT::bytes(int size, int batch) {
            int nr_polarizations = parameters.get_nr_polarizations();
        	return 1ULL * 2 * batch * size * size * nr_polarizations * sizeof(complex<float>);
        }

        double GridFFT::runtime() {
            return compute_runtime(event_start, event_end);
        }

    } // namespace kernel
} // namespace idg
