#include <cstdint> // unint64_t

#include "idg-config.h"
#include "../../common/Kernels.h"
#include "Kernels.h"

//#define COUNT_SINCOS_AS_FLOPS
#if defined(COUNT_SINCOS_AS_FLOPS)
#define FLOPS_PER_SINCOS 8
#endif

namespace idg {
    namespace kernel {
        namespace cpu {

            // Gridder class
            Gridder::Gridder(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_gridder.c_str()),
                parameters(parameters) {}

            void Gridder::run(
                    int nr_subgrids, float w_offset, int nr_channels,
                    void *uvw, void *wavenumbers, void *visibilities,
                    void *spheroidal, void *aterm, void *metadata, void *subgrid) {
                  (sig_gridder (void *) _run)(nr_subgrids, w_offset, nr_channels, uvw, wavenumbers,
                  visibilities, spheroidal, aterm, metadata, subgrid);
            }

            uint64_t Gridder::flops(int nr_baselines, int nr_subgrids) {
                return flops_gridder(parameters, nr_baselines, nr_subgrids);
            }

            uint64_t Gridder::bytes(int nr_baselines, int nr_subgrids) {
                return bytes_gridder(parameters, nr_baselines, nr_subgrids);
            }


            // Degridder class
            Degridder::Degridder(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_degridder.c_str()),
                parameters(parameters) {}

            void Degridder::run(
                    int nr_subgrids, float w_offset, int nr_channels,
                    void *uvw, void *wavenumbers,
                    void *visibilities, void *spheroidal, void *aterm,
                    void *metadata, void *subgrid) {
                  (sig_degridder (void *) _run)(nr_subgrids, w_offset, nr_channels, uvw, wavenumbers,
                  visibilities, spheroidal, aterm, metadata, subgrid);
            }

            uint64_t Degridder::flops(int nr_baselines, int nr_subgrids) {
                return flops_degridder(parameters, nr_baselines, nr_subgrids);
            }

            uint64_t Degridder::bytes(int nr_baselines, int nr_subgrids) {
                return bytes_degridder(parameters, nr_baselines, nr_subgrids);
            }


            // GridFFT class
            GridFFT::GridFFT(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_fft.c_str()),
                parameters(parameters) {}

            void GridFFT::run(int size, int batch, void *data, int direction) {
                (sig_fft (void *) _run)(size, batch, data, direction);
            }

            uint64_t GridFFT::flops(int size, int batch) {
                return flops_fft(parameters, size, batch);
            }

            uint64_t GridFFT::bytes(int size, int batch) {
                return bytes_fft(parameters, size, batch);
            }


            // Adder class
            Adder::Adder(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_adder.c_str()),
                parameters(parameters) {}

            void Adder::run(int nr_subgrids, void *metadata, void *subgrid, void *grid) {
                (sig_adder (void *) _run)(nr_subgrids, metadata, subgrid, grid);
            }

            uint64_t Adder::flops(int nr_subgrids) {
                return flops_adder(parameters, nr_subgrids);
            }

            uint64_t Adder::bytes(int nr_subgrids) {
                return bytes_adder(parameters, nr_subgrids);
            }


            // Splitter class
            Splitter::Splitter(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_splitter.c_str()),
                parameters(parameters) {}

            void Splitter::run(int nr_subgrids, void *metadata, void *subgrid, void *grid) {
                (sig_splitter (void *) _run)(nr_subgrids, metadata, subgrid, grid);
            }

            uint64_t Splitter::flops(int nr_subgrids) {
                return flops_splitter(parameters, nr_subgrids);
            }

            uint64_t Splitter::bytes(int nr_subgrids) {
                return bytes_splitter(parameters, nr_subgrids);
            }

        } // namespace cpu
    } // namespace kernel
} // namespace idg
