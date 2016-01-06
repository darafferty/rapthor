#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

namespace idg {
    namespace kernel {
        namespace cuda {

        // Gridder class
        Gridder::Gridder(cu::Module &module, const Parameters &params) :
            function(module, name_gridder.c_str()), parameters(params) { }

        // Degridder class
        Degridder::Degridder(cu::Module &module, const Parameters &params) :
            function(module, name_degridder.c_str()), parameters(params) { }


        // GridFFT class
        GridFFT::GridFFT(cu::Module &module, const Parameters &params) :
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
            if ((fft_bulk == NULL ||
                size != planned_size) &&
                batch > bulk_size)
            {
                fft_bulk = new cufft::C2C_2D(size, size, stride, dist, bulk_size * nr_polarizations);
            }

            // Plan remainder fft
            if (fft_remainder == NULL ||
                size != planned_size ||
                batch != planned_batch)
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

        // Adder class
        Adder::Adder(cu::Module &module, const Parameters &params) :
            function(module, name_adder.c_str()), parameters(params) { }


        // Splitter class
        Splitter::Splitter(cu::Module &module, const Parameters &params) :
            function(module, name_splitter.c_str()), parameters(params) { }

        // Scaler class
        Scaler::Scaler(cu::Module &module, const Parameters &params) :
            function(module, name_scaler.c_str()), parameters(params) { }

        } // namespace cuda
    } // namespace kernel
} // namespace idg
