#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

namespace idg {

  namespace kernel {

    // Gridder class
    Gridder::Gridder(cu::Module &module, Parameters &parameters) :
        function(module, name_gridder.c_str()),
        parameters(parameters) {}
    
    void Gridder::launchAsync(
        cu::Stream &stream, int jobsize, float w_offset,
        cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
        cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
        cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
        cu::DeviceMemory &d_subgrid) {
        const void *parameters[] = {
            &jobsize, &w_offset, d_uvw, d_wavenumbers, d_visibilities,
            d_spheroidal, d_aterm, d_metadata, d_subgrid };
        int worksize = 16;
        stream.launchKernel(function, jobsize/worksize, 1, 1, 8, 8, 1, 0, parameters);
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
        (nr_polarizations * sizeof(cuFloatComplex) + sizeof(float)) +
        // ATerm
        ((2 * sizeof(int)) + (2 * nr_polarizations * sizeof(cuFloatComplex))) +
        // Spheroidal
    	nr_polarizations * sizeof(cuFloatComplex));
    }


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
        2 * subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex) +
        // UV grid
        subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex) +
        // Visibilities
        nr_time * nr_channels * nr_polarizations * sizeof(cuFloatComplex));
    }

    // GridFFT class
    GridFFT::GridFFT(Parameters &parameters) : parameters(parameters) {
        fft_bulk = NULL;
        fft_remainder = NULL;
    }
    
    void GridFFT::plan(int size, int batch) {
        // Parameters
        int stride = 1;
        int dist = size * size;
        int nr_polarizations = parameters.get_nr_polarizations();
       
        // Plan bulk fft
        if (fft_bulk == NULL ||
            size != planned_size) {
            fft_bulk = new cufft::C2C_2D(size, size, stride, dist, bulk_size * nr_polarizations);
        }

        // Plan remainder fft
        if (fft_remainder == NULL ||
            size != planned_size ||
            batch != planned_batch ||
            size < bulk_size) {
            int remainder = batch % bulk_size;
            fft_remainder = new cufft::C2C_2D(size, size, stride, dist, remainder * nr_polarizations);
        }

        // Set parameters
        planned_size = size;
        planned_batch = batch;
    }
   
    void GridFFT::launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction) {
        // Initialize
        cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(data));
        int s = 0;
        int nr_polarizations = parameters.get_nr_polarizations();

        // Execute bulk ffts
        (*fft_bulk).setStream(stream);
        for (; s < (planned_batch - bulk_size); s += bulk_size) {
            (*fft_bulk).execute(data_ptr, data_ptr, direction);
            data_ptr += bulk_size * planned_size * planned_size * nr_polarizations;
        }

        // Execute remainder ffts
        (*fft_remainder).setStream(stream);
        (*fft_remainder).execute(data_ptr, data_ptr, direction);
    }
    
    uint64_t GridFFT::flops(int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
    }
    
    uint64_t GridFFT::bytes(int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
    	return 1ULL * 2 * batch * size * size * nr_polarizations * sizeof(cuFloatComplex);
    }


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
        1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex);
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
        1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex);
    }

  } // namespace kernel

} // namespace idg
