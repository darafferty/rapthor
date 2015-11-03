#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

#define USE_CUFFT 0

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
            &w_offset, d_uvw, d_wavenumbers, d_visibilities,
            d_spheroidal, d_aterm, d_metadata, d_subgrid };
        stream.launchKernel(function, jobsize, 1, 1, 8, 8, 1, 0, parameters);
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
            &w_offset, d_uvw, d_wavenumbers, d_visibilities,
            d_spheroidal, d_aterm, d_metadata, d_subgrid };
        // IF blockDim.x IS MODIFIED, ALSO MODIFY NR_THREADS IN KernelDegridder.cu
    	stream.launchKernel(function, jobsize, 1, 1, 256, 1, 1, 0, parameters);
    }


    // GridFFT class
    GridFFT::GridFFT(cu::Module &module, Parameters &parameters) :
        parameters(parameters),
        function(module, name_fft.c_str()) {
        fft_bulk = NULL;
        fft_remainder = NULL;
    }

    void GridFFT::plan(int size, int batch) {
        if (size != 32 || USE_CUFFT) {
            // Parameters
            int stride = 1;
            int dist = size * size;
            int nr_polarizations = parameters.get_nr_polarizations();

            // Plan bulk fft
            if ((fft_bulk == NULL ||
                size != planned_size) &&
                batch > bulk_size) {
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
        }

        // Set parameters
        planned_size = size;
        planned_batch = batch;
    }

    void GridFFT::launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction) {
        if (planned_size != 32 || USE_CUFFT) {
            // Initialize
            cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(data));
            int s = 0;
            int nr_polarizations = parameters.get_nr_polarizations();

            // Execute bulk ffts (if any)
            if (planned_batch > bulk_size) {
                (*fft_bulk).setStream(stream);
                for (; s < (planned_batch - bulk_size); s += bulk_size) {
                    (*fft_bulk).execute(data_ptr, data_ptr, direction);
                    data_ptr += bulk_size * planned_size * planned_size * nr_polarizations;
                }
            }

            // Execute remainder ffts
            (*fft_remainder).setStream(stream);
            (*fft_remainder).execute(data_ptr, data_ptr, direction);
        } else {
            cuFloatComplex *data_ptr = reinterpret_cast<cuFloatComplex *>(static_cast<CUdeviceptr>(data));
            int nr_polarizations = parameters.get_nr_polarizations();
            const void *parameters[] = { &data_ptr, &data_ptr, &direction};
            stream.launchKernel(function, planned_batch * nr_polarizations, 1, 1, 128, 1, 1, 0, parameters);
        }
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
