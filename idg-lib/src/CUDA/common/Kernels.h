#ifndef IDG_KERNELS_CUDA_H_
#define IDG_KERNELS_CUDA_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "CU.h"
#include "CUFFT.h"
#include "../../common/Parameters.h"
#include "../../common/Kernels.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            // Kernel names
            static const std::string name_gridder   = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_adder     = "kernel_adder";
            static const std::string name_splitter  = "kernel_splitter";
            static const std::string name_fft       = "kernel_fft";
            static const std::string name_scaler    = "kernel_scaler";

            class Gridder {
                public:
                    Gridder(
                        cu::Module &module,
                        const Parameters &params,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        float w_offset,
                        int nr_channels,
                        cu::DeviceMemory &d_uvw,
                        cu::DeviceMemory &d_wavenumbers,
                        cu::DeviceMemory &d_visibilities,
                        cu::DeviceMemory &d_spheroidal,
                        cu::DeviceMemory &d_aterm,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid) {

                        const void *parameters[] = {
                            &w_offset, &nr_channels, d_uvw, d_wavenumbers, d_visibilities,
                            d_spheroidal, d_aterm, d_metadata, d_subgrid };

                        dim3 grid(nr_subgrids);
                        stream.launchKernel(function, grid, block, 0, parameters);
                    }

                    uint64_t flops(int nr_baselines, int nr_subgrids) {
                        return idg::kernel::flops_gridder(parameters, nr_baselines, nr_subgrids);
                    }

                    uint64_t bytes(int nr_baselines, int nr_subgrids) {
                        return idg::kernel::bytes_gridder(parameters, nr_baselines, nr_subgrids);
                    }

                private:
                    cu::Function function;
                    Parameters parameters;
                    dim3 block;
            };


            class Degridder {
                public:
                    Degridder(
                        cu::Module &module,
                        const Parameters &params,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        float w_offset,
                        int nr_channels,
                        cu::DeviceMemory &d_uvw,
                        cu::DeviceMemory &d_wavenumbers,
                        cu::DeviceMemory &d_visibilities,
                        cu::DeviceMemory &d_spheroidal,
                        cu::DeviceMemory &d_aterm,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid) {

                        const void *parameters[] = {
                            &w_offset, &nr_channels, d_uvw, d_wavenumbers, d_visibilities,
                            d_spheroidal, d_aterm, d_metadata, d_subgrid };

                        dim3 grid(nr_subgrids);
                        stream.launchKernel( function, grid, block, 0, parameters);
                    }

                    uint64_t flops(int nr_baselines, int nr_subgrids) {
                        return idg::kernel::flops_degridder(parameters, nr_baselines, nr_subgrids);
                    }

                    uint64_t bytes(int nr_baselines, int nr_subgrids) {
                        return idg::kernel::bytes_degridder(parameters, nr_baselines, nr_subgrids);
                    }

                private:
                    cu::Function function;
                    Parameters parameters;
                    dim3 block;
            };


            class GridFFT {
                public:
                    GridFFT(
                        cu::Module &module,
                        const Parameters &params);

                    void plan(int size, int batch);

                    void launch(
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

                    void shift(std::complex<float> *data);

                    void scale(std::complex<float> *data, std::complex<float> scale);

                    uint64_t flops(int size, int batch) {
                        return idg::kernel::flops_fft(parameters, size, batch);
                    }

                    uint64_t bytes(int size, int batch) {
                        return idg::kernel::bytes_fft(parameters, size, batch);
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
                    Adder(
                        cu::Module &module,
                        const Parameters &params,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream, int nr_subgrids,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid,
                        cu::DeviceMemory &d_grid) {
                        const void *parameters[] = { d_metadata, d_subgrid, d_grid };
                        dim3 grid(nr_subgrids);
                        stream.launchKernel(function, grid, block, 0, parameters);
                    }

                    uint64_t flops(int nr_subgrids) {
                        return idg::kernel::flops_adder(parameters, nr_subgrids);
                    }

                    uint64_t bytes(int nr_subgrids) {
                        return idg::kernel::bytes_adder(parameters, nr_subgrids);
                    }

                private:
                    cu::Function function;
                    Parameters parameters;
                    dim3 block;
            };


            /*
                Splitter
            */
            class Splitter {
                public:
                    Splitter(
                        cu::Module &module,
                        const Parameters &params,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream, int nr_subgrids,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid,
                        cu::DeviceMemory &d_grid) {
                        const void *parameters[] = { d_metadata, d_subgrid, d_grid };
                        dim3 grid(nr_subgrids);
                        stream.launchKernel(function, grid, block, 0, parameters);
                    }

                    uint64_t flops(int nr_subgrids) {
                        return idg::kernel::flops_splitter(parameters, nr_subgrids);
                    }

                    uint64_t bytes(int nr_subgrids) {
                        return idg::kernel::bytes_splitter(parameters, nr_subgrids);
                    }

                private:
                    cu::Function function;
                    Parameters parameters;
                    dim3 block;
            };


            /*
                Scaler
            */
            class Scaler {
                public:
                    Scaler(
                        cu::Module &module,
                        const Parameters &params,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        cu::DeviceMemory &d_subgrid) {
                        const void *parameters[] = { d_subgrid };
                        dim3 grid(nr_subgrids);
                        stream.launchKernel(function, grid, block, 0, parameters);
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
                    dim3 block;
            };
        } // namespace cuda
    } // namespace kernel
} // namespace idg
#endif
