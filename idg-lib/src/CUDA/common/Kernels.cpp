#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            // Gridder class
            Gridder::Gridder(
                cu::Module &module,
                const Parameters &params,
                const dim3 block) :
                function(module, name_gridder.c_str()), parameters(params), block(block) { }

            void Gridder::launch(
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

            // Degridder class
            Degridder::Degridder(
                cu::Module &module,
                const Parameters &params,
                const dim3 block) :
                function(module, name_degridder.c_str()), parameters(params), block(block) { }

            void Degridder::launch(
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



            // GridFFT class
            GridFFT::GridFFT(
                cu::Module &module,
                const Parameters &params) :
                function(module, name_fft.c_str()), parameters(params)
            {
                fft_bulk = NULL;
                fft_remainder = NULL;
            }

            void GridFFT::plan(int size, int batch) {
                // Parameters
                int stride = 1;
                int dist = size * size;
                int nr_polarizations = parameters.get_nr_polarizations();

                // Plan bulk fft
                if ((fft_bulk == NULL || size != planned_size) && batch > bulk_size)
                {
                    fft_bulk = new cufft::C2C_2D(size, size, stride, dist, bulk_size * nr_polarizations);
                }

                // Plan remainder fft
                if (fft_remainder == NULL || size != planned_size || batch != planned_batch)
                {
                    int remainder = batch % bulk_size;
                    if (remainder > 0) {
                        fft_remainder = new cufft::C2C_2D(size, size, stride, dist, remainder * nr_polarizations);
                    }
                }

                // Set parameters
                planned_size = size;
                planned_batch = batch;
            }

            void GridFFT::launch(
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


            void GridFFT::shift(std::complex<float> *data) {
                int gridsize = parameters.get_grid_size();
                int nr_polarizations = parameters.get_nr_polarizations();

                std::complex<float> tmp13, tmp24;

                // Dimensions
                int n = gridsize;
                int n2 = n / 2;

                // Pointer
                typedef std::complex<float> GridType[nr_polarizations][gridsize][gridsize];
                GridType *x = (GridType *) data;

                // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
                #pragma omp parallel for
                for (int pol = 0; pol < nr_polarizations; pol++) {
                    for (int i = 0; i < n2; i++) {
                        for (int k = 0; k < n2; k++) {
                            tmp13                 = (*x)[pol][i][k];
                            (*x)[pol][i][k]       = (*x)[pol][i+n2][k+n2];
                            (*x)[pol][i+n2][k+n2] = tmp13;

                            tmp24              = (*x)[pol][i+n2][k];
                            (*x)[pol][i+n2][k] = (*x)[pol][i][k+n2];
                            (*x)[pol][i][k+n2] = tmp24;
                         }
                    }
                }
            }

            void GridFFT::scale(std::complex<float> *data, std::complex<float> scale) {
                int gridsize = parameters.get_grid_size();
                int nr_polarizations = parameters.get_nr_polarizations();

                // Pointer
                typedef std::complex<float> GridType[nr_polarizations][gridsize][gridsize];
                GridType *x = (GridType *) data;

                #pragma omp parallel for collapse(2)
                for (int pol = 0; pol < nr_polarizations; pol++) {
                    for (int i = 0; i < gridsize * gridsize; i++) {
                        std::complex<float> value = (*x)[pol][0][i];
                        (*x)[pol][0][i] = std::complex<float>(
                            value.real() * scale.real(),
                            value.imag() * scale.imag());
                    }
                }
            }

            // Adder class
            Adder::Adder(
                cu::Module &module,
                const Parameters &params,
                const dim3 block) :
                function(module, name_adder.c_str()), parameters(params), block(block) {}

            void Adder::launch(
                cu::Stream &stream, int nr_subgrids,
                cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid,
                cu::DeviceMemory &d_grid) {
                const void *parameters[] = { d_metadata, d_subgrid, d_grid };
                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }


            // Splitter class
            Splitter::Splitter(
                cu::Module &module,
                const Parameters &params,
                const dim3 block) :
                function(module, name_splitter.c_str()), parameters(params), block(block) {}

            void Splitter::launch(
                cu::Stream &stream, int nr_subgrids,
                cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid,
                cu::DeviceMemory &d_grid) {
                const void *parameters[] = { d_metadata, d_subgrid, d_grid };
                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }


            // Scaler class
            Scaler::Scaler(
                cu::Module &module,
                const Parameters &params,
                const dim3 block) :
                function(module, name_scaler.c_str()), parameters(params), block(block) {}

            void Scaler::launch(
                cu::Stream &stream,
                int nr_subgrids,
                cu::DeviceMemory &d_subgrid) {
                const void *parameters[] = { d_subgrid };
                dim3 grid(nr_subgrids);
                stream.launchKernel(function, grid, block, 0, parameters);
            }

        } // namespace cuda
    } // namespace kernel
} // namespace idg
