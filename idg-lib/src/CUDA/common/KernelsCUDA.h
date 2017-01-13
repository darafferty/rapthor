#ifndef IDG_KERNELS_CUDA_H_
#define IDG_KERNELS_CUDA_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "CU.h"
#include "CUFFT.h"

#include "idg-common.h"

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
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cu::DeviceMemory &d_uvw,
                        cu::DeviceMemory &d_wavenumbers,
                        cu::DeviceMemory &d_visibilities,
                        cu::DeviceMemory &d_spheroidal,
                        cu::DeviceMemory &d_aterm,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            class Degridder {
                public:
                    Degridder(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cu::DeviceMemory &d_uvw,
                        cu::DeviceMemory &d_wavenumbers,
                        cu::DeviceMemory &d_visibilities,
                        cu::DeviceMemory &d_spheroidal,
                        cu::DeviceMemory &d_aterm,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            class GridFFT {
                public:
                    GridFFT(
                        unsigned int nr_correlations,
                        unsigned int size,
                        cu::Module &module);

                    ~GridFFT();

                    void plan(
                        unsigned int batch);

                    void launch(cu::Stream &stream, cu::DeviceMemory &data, int direction);

                    void shift(std::complex<float> *data);

                    void scale(std::complex<float> *data, std::complex<float> scale);

                private:
                    void plan_bulk();

                private:
                    cu::Function function;
                    unsigned int nr_correlations;
                    unsigned int size;
                    unsigned int planned_batch;
                    const unsigned int bulk_size = 1024;
                    cufft::C2C_2D *fft_bulk;
                    cufft::C2C_2D *fft_remainder;
            };


            class Adder {
                public:
                    Adder(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid,
                        cu::DeviceMemory &d_grid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            /*
                Splitter
            */
            class Splitter {
                public:
                    Splitter(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid,
                        cu::DeviceMemory &d_grid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            /*
                Scaler
            */
            class Scaler {
                public:
                    Scaler(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        cu::DeviceMemory &d_subgrid);

                    uint64_t flops(int nr_subgrids) {
                        //int subgridsize = parameters.get_subgrid_size();
                        //int nr_polarizations = parameters.get_nr_polarizations();
                        //uint64_t flops = 0;
                        //flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // scale
                        //return flops;
                        return 0;
                    }

                    uint64_t bytes(int nr_subgrids) {
                        //int subgridsize = parameters.get_subgrid_size();
                        //int nr_polarizations = parameters.get_nr_polarizations();
                        //uint64_t bytes = 0;
                        //bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2 * sizeof(float); // scale
                        //return bytes;
                        return 0;
                    }

                private:
                    cu::Function function;
                    dim3 block;
            };
        } // namespace cuda
    } // namespace kernel
} // namespace idg
#endif
