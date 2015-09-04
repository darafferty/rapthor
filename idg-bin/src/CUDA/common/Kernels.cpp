#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

namespace idg {

  namespace kernel {

    // Gridder class
    Gridder::Gridder(cu::Module &module) : function(module, name_gridder.c_str()) {}
    
    void Gridder::launchAsync(
        cu::Stream &stream, int jobsize, float w_offset,
        cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
        cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
        cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
        cu::DeviceMemory &d_subgrid) {
        const void *parameters[] = {
            &w_offset, d_uvw, d_wavenumbers, d_visibilities,
            d_spheroidal, d_aterm, d_metadata, d_subgrid };
        printf("%s\n", __func__);
        int worksize = 128;
        stream.launchKernel(function, jobsize/worksize, 1, 1, 8, 8, 1, 0, parameters);
    }   
    
    uint64_t Gridder::flops(int jobsize) {
        return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (
        // LMN
        14 +
        // Phase
        NR_TIME * 10 +
        // Phasor
        NR_TIME * NR_CHANNELS * 4 +
        // ATerm
        NR_POLARIZATIONS * 32 +
        // Spheroidal
        NR_POLARIZATIONS * 2);
    }
    
    uint64_t Gridder::bytes(int jobsize) {
    	return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE *(
        // Grid
        (NR_POLARIZATIONS * sizeof(cuFloatComplex) + sizeof(float)) +
        // ATerm
        ((2 * sizeof(int)) + (2 * NR_POLARIZATIONS * sizeof(cuFloatComplex))) +
        // Spheroidal
    	NR_POLARIZATIONS * sizeof(cuFloatComplex));
    }


    // Degridder class
    Degridder::Degridder(cu::Module &module) : function(module, name_degridder.c_str()) {}

    void Degridder::launchAsync(
        cu::Stream &stream, int jobsize, float w_offset,
        cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
        cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
        cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
        cu::DeviceMemory &d_subgrid) {
        const void *parameters[] = {
            &w_offset, d_uvw, d_wavenumbers, d_visibilities,
            d_spheroidal, d_aterm, d_metadata, d_subgrid };
    	stream.launchKernel(function, jobsize, 1, 1, 128, 1, 1, 0, parameters);
    }
    
    uint64_t Degridder::flops(int jobsize) {
        return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (
        // ATerm
        NR_POLARIZATIONS * 32 +
        // Spheroidal
        NR_POLARIZATIONS * 2 +
        // LMN
        14 +
        // Phase
        10 +
        // Phasor
        NR_TIME * NR_CHANNELS * 4 +
        // Degrid
        NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * 8);
    }
    
    uint64_t Degridder::bytes(int jobsize) {
        return 1ULL * jobsize * (
        // ATerm
        2 * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex) +
        // UV grid
        SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex) +
        // Visibilities
        NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(cuFloatComplex));
    }

    // GridFFT class
    GridFFT::GridFFT() {
        fft = NULL;
    }
    
    void GridFFT::plan(int size, int batch) {
        // Check wheter a new plan has to be created
        if (fft == NULL ||
            size == planned_size ||
            batch == planned_batch) {

            // Create new plan
            int stride = 1;
            int dist = size * size;
            fft = new cufft::C2C_2D(size, size, stride, dist, batch * NR_POLARIZATIONS);
            
            // Update parameters
            planned_size = size;
            planned_batch = batch;
        }
    }
    
    void GridFFT::launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction) {
        cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(data));
        (*fft).setStream(stream);
        (*fft).execute(data_ptr, data_ptr, direction);
    }
    
    uint64_t GridFFT::flops(int size, int batch) {
    	return 1ULL * batch * NR_POLARIZATIONS * 5 * size * size * log(size * size);
    }
    
    uint64_t GridFFT::bytes(int size, int batch) {
    	return 1ULL * 2 * batch * size * size * NR_POLARIZATIONS * sizeof(cuFloatComplex);
    }


    // Adder class
    Adder::Adder(cu::Module &module) : function(module, name_adder.c_str()) {}
    
    void Adder::launchAsync(
    	cu::Stream &stream, int jobsize,
    	cu::DeviceMemory &d_metadata,
    	cu::DeviceMemory &d_subgrid,
    	cu::DeviceMemory &d_grid) {
    	const void *parameters[] = { &jobsize, d_metadata, d_subgrid, d_grid };
    	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
    }
    
    uint64_t Adder::flops(int jobsize) {
    	return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2;
    }
    
    uint64_t Adder::bytes(int jobsize) {
    	return
        // Coordinate
        1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * 2 * sizeof(int) +
        // Grid
        1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex);
    }

   
    // Splitter class
    Splitter::Splitter(cu::Module &module) : function(module, name_splitter.c_str()) {}
    
    void Splitter::launchAsync(
     	cu::Stream &stream, int jobsize,
    	cu::DeviceMemory &d_metadata,
    	cu::DeviceMemory &d_subgrid,
    	cu::DeviceMemory &d_grid) {
    	const void *parameters[] = { &jobsize, d_metadata, d_subgrid, d_grid };
    	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
    }
    
    uint64_t Splitter::flops(int jobsize) {
    	return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2;
    }
    
    uint64_t Splitter::bytes(int jobsize) {
    	return
        // Coordinate
        1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * 2 * sizeof(int) +
        // Grid
        1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex);
    }

  } // namespace kernel

} // namespace idg
