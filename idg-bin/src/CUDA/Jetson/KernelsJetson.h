#ifndef IDG_KERNELS_CUDA_JETSON_H_
#define IDG_KERNELS_CUDA_JETSON_H_

#include "../common/Kernels.h"


namespace idg {
    namespace kernel {
        namespace cuda {

            class GridderJetson : public Gridder {
            public:

                GridderJetson(cu::Module& module, const Parameters& params)
                    : Gridder(module, params)
                    {}

                static const int max_nr_timesteps = 32;

                virtual void launch(
                    cu::Stream &stream,
                    int nr_baselines,
                    float w_offset,
                    cu::DeviceMemory &d_uvw,
                    cu::DeviceMemory &d_wavenumbers,
                    cu::DeviceMemory &d_visibilities,
                    cu::DeviceMemory &d_spheroidal,
                    cu::DeviceMemory &d_aterm,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid) override
                    {
                        launchAsync<32,4,1>(
                            stream,
                            nr_baselines,
                            w_offset,
                            d_uvw,
                            d_wavenumbers,
                            d_visibilities,
                            d_spheroidal,
                            d_aterm,
                            d_metadata,
                            d_subgrid);
                    }

                virtual int get_max_nr_timesteps() {
                    return max_nr_timesteps;
                }
            };


            class DegridderJetson : public Degridder {
            public:
                DegridderJetson(cu::Module& module, const Parameters& params)
                    : Degridder(module, params)
                    {}

                static const int max_nr_timesteps = 64;
                static const int nr_threads = 128;

                virtual void launch(
                    cu::Stream &stream,
                    int nr_baselines,
                    float w_offset,
                    cu::DeviceMemory &d_uvw,
                    cu::DeviceMemory &d_wavenumbers,
                    cu::DeviceMemory &d_visibilities,
                    cu::DeviceMemory &d_spheroidal,
                    cu::DeviceMemory &d_aterm,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid) override
                    {
                        launchAsync<nr_threads,1,1>(
                        stream,
                        nr_baselines,
                        w_offset,
                        d_uvw,
                        d_wavenumbers,
                        d_visibilities,
                        d_spheroidal,
                        d_aterm,
                        d_metadata,
                        d_subgrid);
                    }

                virtual int get_max_nr_timesteps() {
                    return max_nr_timesteps;
                }
            };


            class GridFFTJetson : public GridFFT {
            public:
                GridFFTJetson(cu::Module& module, const Parameters& params)
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


            class AdderJetson : public Adder {
            public:
                AdderJetson(cu::Module& module, const Parameters& params)
                    : Adder(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream, int nr_subgrids,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid)
                {
                    launchAsync<128, 1, 1>(stream, nr_subgrids, d_metadata, d_subgrid, d_grid);
                }
            };


            class SplitterJetson : public Splitter {
            public:
                SplitterJetson(cu::Module& module, const Parameters& params)
                    : Splitter(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream, int nr_subgrids,
                    cu::DeviceMemory &d_metadata,
                    cu::DeviceMemory &d_subgrid,
                    cu::DeviceMemory &d_grid)
                {
                    launchAsync<128, 1, 1>(stream, nr_subgrids, d_metadata, d_subgrid, d_grid);
                }
            };


            class ScalerJetson : public Scaler {
            public:
                ScalerJetson(cu::Module& module, const Parameters& params)
                    : Scaler(module, params)
                {}

                virtual void launch(
                    cu::Stream &stream, int nr_subgrids,
                    cu::DeviceMemory &d_subgrid) override
                {
                    launchAsync<128,1,1>(stream, nr_subgrids, d_subgrid);
                }
            };

        } // namespace cuda
    } // namespace kernel
} // namespace idg
#endif
