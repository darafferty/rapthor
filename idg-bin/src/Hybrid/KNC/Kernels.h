#ifndef IDG_OFFLOAD_KERNELS_H_
#define IDG_OFFLOAD_KERNELS_H_

#pragma omp declare target

#include <cstdint>

#include "../../common/Parameters.h"

namespace idg {
    namespace kernel {
        namespace knc {

            void kernel_gridder (
                const int jobsize, const float w_offset,
                const void *uvw,
                const void *wavenumbers,
                const void *visibilities,
                const void *spheroidal,
                const void *aterm,
                const void *metadata,
                      void *subgrid,
                const int nr_stations,
                const int nr_timesteps,
                const int nr_timeslots,
                const int nr_channels,
                const int gridsize,
                const int subgridsize,
                const float imagesize,
                const int nr_polarizations
                );

            void kernel_degridder(
                const int jobsize, const float w_offset,
                const void *_uvw,
                const void *_wavenumbers,
                      void *_visibilities,
                const void *_spheroidal,
                const void *_aterm,
                const void *_metadata,
                const void *_subgrid,
                const int nr_stations,
                const int nr_timesteps,
                const int nr_timeslots,
                const int nr_channels,
                const int gridsize,
                const int subgridsize,
                const float imagesize,
                const int nr_polarizations
                );

            void kernel_fft(
                const int size,
                const int batch,
                void *data,
                const int sign,
                const int nr_polarizations
                );

            void kernel_adder(
                const int jobsize,
                const void *_metadata,
                const void *_subgrid,
                      void *grid,
                const int gridsize,
                const int subgridsize,
                const int nr_polarizations);

            void kernel_splitter(
                const int jobsize,
                const void *metadata,
                      void *subgrid,
                const void *grid,
                const int gridsize,
                const int subgridsize,
                const int nr_polarizations);

#if 0
            uint64_t kernel_gridder_flops(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids);

            uint64_t kernel_gridder_bytes(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids);

            uint64_t kernel_degridder_flops(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids);

            uint64_t kernel_degridder_bytes(
                const Parameters &parameters,
                int jobsize,
                int nr_subgrids);

            uint64_t kernel_fft_flops(
                const Parameters &parameters,
                int size,
                int batch);

            uint64_t kernel_fft_bytes(
                const Parameters &parameters,
                int size,
                int batch);

            uint64_t kernel_adder_flops(
                const Parameters &parameters,
                int jobsize);

            uint64_t kernel_adder_bytes(
                const Parameters &parameters,
                int jobsize);

            uint64_t kernel_splitter_flops(
                const Parameters &parameters,
                int jobsize);

            uint64_t kernel_splitter_bytes(
                const Parameters &parameters,
                int jobsize);
#endif

        } // namespace knc
    } // namespace kernel
} // namespace idg

#pragma omp end declare target
#endif
