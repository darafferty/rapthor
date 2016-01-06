#ifndef IDG_KERNELS_CUDA_MAXWELL_H_
#define IDG_KERNELS_CUDA_MAXWELL_H_

#include "../common/Kernels.h"


namespace idg {
    namespace kernel {
        namespace cuda {

            class GridderMaxwell : public Gridder {
            public:

                GridderMaxwell(cu::Module& module, const Parameters& params)
                    : Gridder(module, params)
                    {}

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
                    cu::DeviceMemory &d_subgrid) override
                    {
                        launchAsync<8,8,1>(
                            stream,
                            jobsize,
                            w_offset,
                            d_uvw,
                            d_wavenumbers,
                            d_visibilities,
                            d_spheroidal,
                            d_aterm,
                            d_metadata,
                            d_subgrid);
                    }
            };


            class DegridderMaxwell : public Degridder {
            public:
                DegridderMaxwell(cu::Module& module, const Parameters& params)
                    : Degridder(module, params)
                    {}

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
                    cu::DeviceMemory &d_subgrid) override
                    {
                        launchAsync<256,1,1>(
                        stream,
                        jobsize,
                        w_offset,
                        d_uvw,
                        d_wavenumbers,
                        d_visibilities,
                        d_spheroidal,
                        d_aterm,
                        d_metadata,
                        d_subgrid);
                    }
            };


            class GridFFTMaxwell : public GridFFT {
            public:
                GridFFTMaxwell(cu::Module& module, const Parameters& params)
                    : GridFFT(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream,
                    cu::DeviceMemory &data,
                    int direction) override
                {
                    launchAsync<128,1,1>(stream, data, direction);
                }
            };


            class AdderMaxwell : public Adder {
            public:
                AdderMaxwell(cu::Module& module, const Parameters& params)
                    : Adder(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid)
                {
                    launchAsync<128, 1, 1>(stream, jobsize, d_metadata, d_subgrid, d_grid);
                }
            };


            class SplitterMaxwell : public Splitter {
            public:
                SplitterMaxwell(cu::Module& module, const Parameters& params)
                    : Splitter(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid)
                {
                    launchAsync<128, 1, 1>(stream, jobsize, d_metadata, d_subgrid, d_grid);
                }
            };


            class ScalerMaxwell : public Scaler {
            public:
                ScalerMaxwell(cu::Module& module, const Parameters& params)
                    : Scaler(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream, int jobsize,
                    cu::DeviceMemory &d_subgrid) override
                {
                    launchAsync<128,1,1>(stream, jobsize, d_subgrid);
                }
            };
        } // namespace cuda
    } // namespace kernel
} // namespace idg
#endif
