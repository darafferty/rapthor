#ifndef IDG_KERNELS_CUDA_H_
#define IDG_KERNELS_CUDA_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "CU.h"
#include "CUFFT.h"
#include "../../common/Parameters.h"


namespace idg {
    namespace kernel {
        namespace cuda {

            // define the kernel function names
            static const std::string name_gridder   = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_adder     = "kernel_adder";
            static const std::string name_splitter  = "kernel_splitter";
            static const std::string name_fft       = "kernel_fft";
            static const std::string name_scaler    = "kernel_scaler";

            class Gridder {
            public:

                Gridder(cu::Module &module, const Parameters &params);

                virtual void launch(
                    cu::Stream &stream,
                    int jobsize,
                    float w_offset,
                    cu::DeviceMemory &d_uvw,
                    cu::DeviceMemory &d_wavenumbers,
                    cu::DeviceMemory &d_visibilities,
                    cu::DeviceMemory &d_spheroidal,
                    cu::DeviceMemory &d_aterm,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid) = 0;

                template <int blockX, int blockY, int blockZ>
                void launchAsync(
                    cu::Stream &stream,
                    int jobsize,
                    float w_offset,
                    cu::DeviceMemory &d_uvw,
                    cu::DeviceMemory &d_wavenumbers,
                    cu::DeviceMemory &d_visibilities,
                    cu::DeviceMemory &d_spheroidal,
                    cu::DeviceMemory &d_aterm,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid) {

                    const void *parameters[] = {
                        &w_offset, d_uvw, d_wavenumbers, d_visibilities,
                        d_spheroidal, d_aterm, d_metadata, d_subgrid };

                    stream.launchKernel(function, jobsize, 1, 1,
                                        blockX, blockY, blockZ, 0, parameters);
                }

                uint64_t flops(int jobsize, int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_time = parameters.get_nr_time();
                    int nr_channels = parameters.get_nr_channels();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t flops = 0;
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase index
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase offset
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * 2; // phase
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 6; // shift
                    return flops;
                }

                uint64_t bytes(int jobsize, int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_time = parameters.get_nr_time();
                    int nr_channels = parameters.get_nr_channels();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t bytes = 0;
                    bytes += 1ULL * jobsize * nr_time * 3 * sizeof(float); // uvw
                    bytes += 1ULL * jobsize * nr_time * nr_channels * nr_polarizations * sizeof(cuFloatComplex); // visibilities
                    bytes += 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize  * sizeof(cuFloatComplex); // subgrids
                    return bytes;
                }

        	private:
                cu::Function function;
                Parameters parameters;
            };


            class Degridder {
            public:
                Degridder(cu::Module &module, const Parameters &params);

                virtual void launch(
                    cu::Stream &stream,
                    int jobsize,
                    float w_offset,
                    cu::DeviceMemory &d_uvw,
                    cu::DeviceMemory &d_wavenumbers,
                    cu::DeviceMemory &d_visibilities,
                    cu::DeviceMemory &d_spheroidal,
                    cu::DeviceMemory &d_aterm,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid) = 0;

                template <int blockX, int blockY, int blockZ>
                void launchAsync(
                    cu::Stream &stream,
                    int jobsize,
                    float w_offset,
                    cu::DeviceMemory &d_uvw,
                    cu::DeviceMemory &d_wavenumbers,
                    cu::DeviceMemory &d_visibilities,
                    cu::DeviceMemory &d_spheroidal,
                    cu::DeviceMemory &d_aterm,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid) {

                    const void *parameters[] = {
                        &w_offset, d_uvw, d_wavenumbers, d_visibilities,
                        d_spheroidal, d_aterm, d_metadata, d_subgrid };

                    // IF blockDim.x IS MODIFIED, ALSO MODIFY NR_THREADS IN KernelDegridder.cu
                    stream.launchKernel(function, jobsize, 1, 1,
                                        blockX, blockY, blockZ, 0, parameters);
                }

                uint64_t flops(int jobsize, int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_time = parameters.get_nr_time();
                    int nr_channels = parameters.get_nr_channels();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t flops = 0;
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase index
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase offset
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * 2; // phase
                    flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 6; // shift
                    return flops;
                }

                uint64_t bytes(int jobsize, int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_time = parameters.get_nr_time();
                    int nr_channels = parameters.get_nr_channels();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t bytes = 0;
                    bytes += 1ULL * jobsize * nr_time * 3 * sizeof(float); // uvw
                    bytes += 1ULL * jobsize * nr_time * nr_channels * nr_polarizations * sizeof(cuFloatComplex); // visibilities
                    bytes += 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize  * sizeof(cuFloatComplex); // subgrids
                    return bytes;
                }

        	private:
        	    cu::Function function;
                Parameters parameters;
            };


            class GridFFT {
        	public:
                GridFFT(cu::Module &module, const Parameters &params);

                void plan(int size, int batch);

                virtual void launch(
                    cu::Stream &stream,
                    cu::DeviceMemory &data,
                    int direction) = 0;

                template <int blockX, int blockY, int blockZ>
                void launchAsync(
                    cu::Stream &stream,
                    cu::DeviceMemory &data,
                    int direction)
                {
                    // Initialize
                    cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(data));
                    int s = 0;
                    int nr_polarizations = parameters.get_nr_polarizations();

                    // Execute bulk ffts (if any)
                    if (planned_batch >= bulk_size) {
                        (*fft_bulk).setStream(stream);
                        for (; s < planned_batch; s += bulk_size) {
                            if (planned_batch - s >= bulk_size) {
                                (*fft_bulk).execute(data_ptr, data_ptr, direction);
                                data_ptr += bulk_size * planned_size * planned_size * nr_polarizations;
                            }
                        }
                    }

                    // Execute remainder ffts
                    if (s < planned_batch) {
                        (*fft_remainder).setStream(stream);
                        (*fft_remainder).execute(data_ptr, data_ptr, direction);
                    }

                    // Custom FFT kernel is disabled
                    //cuFloatComplex *data_ptr = reinterpret_cast<cuFloatComplex *>(static_cast<CUdeviceptr>(data));
                    //int nr_polarizations = parameters.get_nr_polarizations();
                    //const void *parameters[] = { &data_ptr, &data_ptr, &direction};
                    //stream.launchKernel(function, planned_batch * nr_polarizations, 1, 1,
                    //                    blockX, blockY, blockZ, 0, parameters);
                }

                uint64_t flops(int size, int batch) {
                    int nr_polarizations = parameters.get_nr_polarizations();
                    return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size) / log(2.0);
                }

                uint64_t bytes(int size, int batch) {
                    int nr_polarizations = parameters.get_nr_polarizations();
                    return 1ULL * 2 * batch * size * size * nr_polarizations * sizeof(cuFloatComplex);
                }

            private:
                cu::Function function;
                Parameters parameters;
                int planned_size;
                int planned_batch;
                const int bulk_size = 1024;
                cufft::C2C_2D *fft_bulk;
                cufft::C2C_2D *fft_remainder;
            };


            class Adder {
            public:
                Adder(cu::Module &module, const Parameters &params);

                virtual void launch(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid) = 0;

                template <int blockX, int blockY, int blockZ>
                void launchAsync(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid) {
                    const void *parameters[] = { d_metadata, d_subgrid, d_grid };
                    stream.launchKernel(function, jobsize, 1, 1,
                                        blockX, blockY, blockZ, 0, parameters);
                }

                uint64_t flops(int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t flops = 0;
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 4; // add
                    return flops;
                }

                uint64_t bytes(int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t bytes = 0;
                    bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
                    bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
                    bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid in
                    bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
                    return bytes;
                }

            private:
                cu::Function function;
                Parameters parameters;
            };


            /*
              Splitter
            */
            class Splitter {
            public:
                Splitter(cu::Module &module, const Parameters &params);

                virtual void launch(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid) = 0;

                template <int blockX, int blockY, int blockZ>
                void launchAsync(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid) {
                    const void *parameters[] = { d_metadata, d_subgrid, d_grid };
                    stream.launchKernel(function, jobsize, 1, 1,
                                        blockX, blockY, blockZ, 0, parameters);
                }

                uint64_t flops(int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    uint64_t flops = 0;
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
                    return flops;
                }

                uint64_t bytes(int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t bytes = 0;
                    bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
                    bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
                    bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
                    return bytes;
                }

            private:
                cu::Function function;
                Parameters parameters;
            };

            class Scaler {
            public:
                Scaler(cu::Module &module, const Parameters &params);

                virtual void launch(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_subgrid) = 0;

                template <int blockX, int blockY, int blockZ>
                void launchAsync(
                    cu::Stream &stream,
                    int jobsize,
                    cu::DeviceMemory &d_subgrid) {

                    const void *parameters[] = { d_subgrid };

                    stream.launchKernel(function, jobsize, 1, 1,
                                        blockX, blockY, blockZ, 0, parameters);
                }

                uint64_t flops(int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t flops = 0;
                    flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // scale
                    return flops;
                }

                uint64_t bytes(int nr_subgrids) {
                    int subgridsize = parameters.get_subgrid_size();
                    int nr_polarizations = parameters.get_nr_polarizations();
                    uint64_t bytes = 0;
                    bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2 * sizeof(float); // scale
                    return bytes;
                }

            private:
                cu::Function function;
                Parameters parameters;
            };
        } // namespace cuda
    } // namespace kernel
} // namespace idg
#endif
