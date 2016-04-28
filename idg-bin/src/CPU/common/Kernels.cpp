#include <cstdint> // unint64_t

#include "idg-config.h"
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

            uint64_t gridder_flops(Parameters &parameters, int nr_baselines, int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();

                // Number of flops per uvw coordinate
                uint64_t flops_per_uvw = 0;
                flops_per_uvw += 1ULL * nr_time * 5; // phase index
                flops_per_uvw += 1ULL * nr_time * 5; // phase offset

                // Number of flops per visibility
                uint64_t flops_per_vis = 0;
                flops_per_vis += 1ULL * nr_time * nr_channels * 2; // phase
                #if defined(COUNT_SINCOS_AS_FLOPS)
                flops_per_vis += 1ULL * nr_time * nr_channels * FLOPS_PER_SINCOS; // phasor
                #endif

                // Number of flops per pixel
                uint64_t flops_per_pix = 0;
                flops_per_pix += 1ULL * nr_baselines * nr_time * nr_channels * (nr_polarizations * 8); // update
                flops_per_pix += 1ULL * nr_subgrids * nr_polarizations * 30; // aterm
                flops_per_pix += 1ULL * nr_subgrids * nr_polarizations * 2; // spheroidal
                flops_per_pix += 1ULL * nr_subgrids * 6; // shift

                // Total number of flops
                uint64_t flops_total = 0;
                flops_total += 1ULL * nr_subgrids * subgridsize * subgridsize * (flops_per_uvw + flops_per_vis);
                flops_total += 1ULL * subgridsize * subgridsize * flops_per_pix;
                return flops_total;
            }

            uint64_t Gridder::flops(int nr_baselines, int nr_subgrids) {
                return gridder_flops(parameters, nr_baselines, nr_subgrids);
            }

            uint64_t gridder_bytes(Parameters &parameters, int nr_baselines, int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();

                // Number of bytes per uvw coordinate
                uint64_t bytes_per_uvw = 0;
                bytes_per_uvw += 1ULL * 3 * sizeof(float); // uvw

                // Number of bytes per visibility
                uint64_t bytes_per_vis = 0;
                bytes_per_vis += 1ULL * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities

                // Number of bytes per pixel
                uint64_t bytes_per_pix = 0;
                bytes_per_pix += 1ULL * nr_polarizations * 2 * sizeof(float); // pixel

                // Total number of bytes
                uint64_t bytes_total = 0;
                bytes_total += 1ULL * nr_baselines * nr_time * bytes_per_uvw;
                bytes_total += 1ULL * nr_baselines * nr_time * bytes_per_vis;
                bytes_total += 1ULL * nr_subgrids * subgridsize * subgridsize * bytes_per_pix;
                return bytes_total;
            }

            uint64_t Gridder::bytes(int nr_baselines, int nr_subgrids) {
                return gridder_bytes(parameters, nr_baselines, nr_subgrids);
            }


            // Degridder class
            Degridder::Degridder(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_degridder.c_str()),
                parameters(parameters) {}

            void Degridder::run(
                    int nr_subgrids, float w_offset, void *uvw, void *wavenumbers,
                    void *visibilities, void *spheroidal, void *aterm,
                    void *metadata, void *subgrid) {
                  (sig_degridder (void *) _run)(nr_subgrids, w_offset, uvw, wavenumbers,
                  visibilities, spheroidal, aterm, metadata, subgrid);
            }

            uint64_t Degridder::flops(int nr_baselines, int nr_subgrids) {
                return gridder_flops(parameters, nr_baselines, nr_subgrids);
            }

            uint64_t Degridder::bytes(int nr_baselines, int nr_subgrids) {
                return gridder_bytes(parameters, nr_baselines, nr_subgrids);
            }


            // GridFFT class
            GridFFT::GridFFT(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_fft.c_str()),
                parameters(parameters) {}

            void GridFFT::run(int size, int batch, void *data, int direction) {
                (sig_fft (void *) _run)(size, batch, data, direction);
            }

            uint64_t GridFFT::flops(int size, int batch) {
                int nr_polarizations = parameters.get_nr_polarizations();
        	    return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
            }

            uint64_t GridFFT::bytes(int size, int batch) {
                int nr_polarizations = parameters.get_nr_polarizations();
        	    return 1ULL * 2 * batch * nr_polarizations * size * size * 2 * sizeof(float);
            }


            // Adder class
            Adder::Adder(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_adder.c_str()),
                parameters(parameters) {}

            void Adder::run(int nr_subgrids, void *metadata, void *subgrid, void *grid) {
                (sig_adder (void *) _run)(nr_subgrids, metadata, subgrid, grid);
            }

            uint64_t Adder::flops(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // add
                return flops;
            }

            uint64_t Adder::bytes(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid in
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
                return bytes;
            }


            // Splitter class
            Splitter::Splitter(runtime::Module &module, const Parameters &parameters) :
                _run(module, name_splitter.c_str()),
                parameters(parameters) {}

            void Splitter::run(int nr_subgrids, void *metadata, void *subgrid, void *grid) {
                (sig_splitter (void *) _run)(nr_subgrids, metadata, subgrid, grid);
            }

            uint64_t Splitter::flops(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                uint64_t flops = 0;
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
                return flops;
            }

            uint64_t Splitter::bytes(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
                return bytes;
            }

        } // namespace cpu
    } // namespace kernel
} // namespace idg
