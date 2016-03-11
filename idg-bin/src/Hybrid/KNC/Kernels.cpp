
#include <cstdint>
#include <cmath>
#include "Kernels.h"

#define COUNT_SINCOS_AS_FLOPS
#if defined(COUNT_SINCOS_AS_FLOPS)
#define FLOPS_PER_SINCOS 8
#endif

namespace idg {
    namespace kernel {
        namespace knc {

            uint64_t kernel_gridder_flops(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids)
            {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase index
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase offset
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * 2; // phase
                #if defined(COUNT_SINCOS_AS_FLOPS)
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * FLOPS_PER_SINCOS; // phasor
                #endif
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }


            uint64_t kernel_gridder_bytes(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids)
            {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * jobsize * nr_time * 3 * sizeof(float); // uvw
                bytes += 1ULL * jobsize * nr_time * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities
                bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * 2 * sizeof(float); // subgrids
                return bytes;
            }


            uint64_t kernel_degridder_flops(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids)
            {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase index
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * 5; // phase offset
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * 2; // phase
                #if defined(COUNT_SINCOS_AS_FLOPS)
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * FLOPS_PER_SINCOS; // phasor
                #endif
                flops += 1ULL * jobsize * nr_time * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }


            uint64_t kernel_degridder_bytes(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids)
            {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * jobsize * nr_time * 3 * sizeof(float); // uvw
                bytes += 1ULL * jobsize * nr_time * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities
                bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * 2 * sizeof(float); // subgrids
                return bytes;
            }

            uint64_t kernel_fft_flops(
                const Parameters &parameters,
                int size,
                int batch)
            {
                int nr_polarizations = parameters.get_nr_polarizations();
                return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
            }


            uint64_t kernel_fft_bytes(
                const Parameters &parameters,
                int size,
                int batch)
            {
                int nr_polarizations = parameters.get_nr_polarizations();
                return 1ULL * 2 * batch * nr_polarizations * size * size * 2 * sizeof(float);
            }


            uint64_t kernel_adder_flops(
                const Parameters &parameters,
                int jobsize)
            {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * jobsize * subgridsize * subgridsize * 8; // shift
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // add
                return flops;
            }


            uint64_t kernel_adder_bytes(
                const Parameters &parameters,
                int jobsize)
            {
                int subgridsize = parameters.get_subgrid_size();
                uint64_t bytes = 0;
                bytes += 1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(unsigned); // coordinate
                bytes += 1ULL * jobsize * subgridsize * subgridsize * 3 * sizeof(unsigned); // pixels
                return bytes;
            }


            uint64_t kernel_splitter_flops(
                const Parameters &parameters,
                int jobsize)
            {
                int subgridsize = parameters.get_subgrid_size();
                uint64_t flops = 0;
                flops += 1ULL * jobsize * subgridsize * subgridsize * 8; // shift
                return flops;
            }


            uint64_t kernel_splitter_bytes(
                const Parameters &parameters,
                int jobsize)
            {
                int subgridsize = parameters.get_subgrid_size();
                uint64_t bytes = 0;
                bytes += 1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(unsigned); // coordinate
                bytes += 1ULL * jobsize * subgridsize * subgridsize * 3 * sizeof(unsigned); // pixels
                return bytes;
            }

        } // namespace knc
    } // namespace kernel
}  // namespace idg
