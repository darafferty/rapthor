#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

using namespace std;

namespace idg {

  namespace kernel {

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
        int wgSize = 8;
        cl::NDRange globalSize(jobsize);
        cl::NDRange localSize(wgSize, wgSize);
        kernel.setArg(0, w_offset);
        kernel.setArg(1, d_uvw);
        kernel.setArg(2, d_wavenumbers);
        kernel.setArg(3, d_visibilities);
        kernel.setArg(4, d_spheroidal);
        kernel.setArg(5, d_aterm);
        kernel.setArg(6, d_metadata);
        kernel.setArg(7, d_subgrid);
        try {
            cl::Event event;
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
        } catch (cl::Error &error) {
            std::cerr << "Error launching gridder: " << error.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }   
    
    uint64_t Gridder::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_time = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        return 1ULL * jobsize * subgridsize * subgridsize * (
        // LMN
        14 +
        // Phase
        nr_time * 10 +
        // Phasor
        nr_time * nr_channels * 4 +
        // ATerm
        nr_polarizations * 32 +
        // Spheroidal
        nr_polarizations * 2);
    }
    
    uint64_t Gridder::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_time = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * jobsize * subgridsize * subgridsize *(
        // Grid
        (nr_polarizations * sizeof(complex<float>) + sizeof(float)) +
        // ATerm
        ((2 * sizeof(int)) + (2 * nr_polarizations * sizeof(complex<float>))) +
        // Spheroidal
    	nr_polarizations * sizeof(complex<float>));
    }

#if 0
    // Degridder class
    Degridder::Degridder(cu::Module &module, Parameters &parameters) :
        function(module, name_degridder.c_str()),
        parameters(parameters) {}

    void Degridder::launchAsync(
        cu::Stream &stream, int jobsize, float w_offset,
        cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
        cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
        cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
        cu::DeviceMemory &d_subgrid) {
        const void *parameters[] = {
            &jobsize, &w_offset, d_uvw, d_wavenumbers, d_visibilities,
            d_spheroidal, d_aterm, d_metadata, d_subgrid };
        int worksize = 16;
    	stream.launchKernel(function, jobsize/worksize, 1, 1, 128, 1, 1, 0, parameters);
    }
    
    uint64_t Degridder::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_time = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        return 1ULL * jobsize * subgridsize * subgridsize * (
        // ATerm
        nr_polarizations * 32 +
        // Spheroidal
        nr_polarizations * 2 +
        // LMN
        14 +
        // Phase
        10 +
        // Phasor
        nr_time * nr_channels * 4 +
        // Degrid
        nr_time * nr_channels * nr_polarizations * 8);
    }
    
    uint64_t Degridder::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_time = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        return 1ULL * jobsize * (
        // ATerm
        2 * subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>) +
        // UV grid
        subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>) +
        // Visibilities
        nr_time * nr_channels * nr_polarizations * sizeof(complex<float>));
    }
#endif

    // GridFFT class
    GridFFT::GridFFT(Parameters &parameters) : parameters(parameters) {
    }
    
    void GridFFT::plan(cl::Context &context, int size, int batch) {
        // Set parameters
        planned_size = size;
        planned_batch = batch;
    }
   
    void GridFFT::launchAsync(
        cl::CommandQueue &queue, cl::Buffer &d_data, int direction) {
    }
    
    uint64_t GridFFT::flops(int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
    }
    
    uint64_t GridFFT::bytes(int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * 2 * batch * size * size * nr_polarizations * sizeof(complex<float>);
    }


#if 0
    // Adder class
    Adder::Adder(cu::Module &module, Parameters &parameters) :
        function(module, name_adder.c_str()),
        parameters(parameters) {}
    
    void Adder::launchAsync(
    	cu::Stream &stream, int jobsize,
    	cu::DeviceMemory &d_metadata,
    	cu::DeviceMemory &d_subgrid,
    	cu::DeviceMemory &d_grid) {
    	const void *parameters[] = { &jobsize, d_metadata, d_subgrid, d_grid };
    	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
    }
    
    uint64_t Adder::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2;
    }
    
    uint64_t Adder::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
    	return
        // Coordinate
        1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(int) +
        // Grid
        1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>);
    }

   
    // Splitter class
    Splitter::Splitter(cu::Module &module, Parameters &parameters) :
        function(module, name_splitter.c_str()),
        parameters(parameters) {}
    
    void Splitter::launchAsync(
     	cu::Stream &stream, int jobsize,
    	cu::DeviceMemory &d_metadata,
    	cu::DeviceMemory &d_subgrid,
    	cu::DeviceMemory &d_grid) {
    	const void *parameters[] = { &jobsize, d_metadata, d_subgrid, d_grid };
    	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
    }
    
    uint64_t Splitter::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2;
    }
    
    uint64_t Splitter::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
    	return
        // Coordinate
        1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(int) +
        // Grid
        1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(complex<float>);
    }

#endif
  } // namespace kernel

} // namespace idg
