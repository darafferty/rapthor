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

        // Degridder class
        Degridder::Degridder(
            cu::Module &module,
            const Parameters &params,
            const dim3 block) :
            function(module, name_degridder.c_str()), parameters(params), block(block) { }


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

        // Splitter class
        Splitter::Splitter(
            cu::Module &module,
            const Parameters &params,
            const dim3 block) :
            function(module, name_splitter.c_str()), parameters(params), block(block) {}

        // Scaler class
        Scaler::Scaler(
            cu::Module &module,
            const Parameters &params,
            const dim3 block) :
            function(module, name_scaler.c_str()), parameters(params), block(block) {}

        } // namespace cuda
    } // namespace kernel
} // namespace idg
