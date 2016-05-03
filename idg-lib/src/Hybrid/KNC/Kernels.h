#ifndef IDG_OFFLOAD_KERNELS_H_
#define IDG_OFFLOAD_KERNELS_H_

#pragma omp declare target

#include <cstdint>

#include "../../common/Parameters.h"

namespace idg {
    namespace kernel {
        namespace knc {

            void gridder (
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

            void degridder(
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

            void fft(
                const int size,
                const int batch,
                void *data,
                const int sign,
                const int nr_polarizations
                );

            void ifftshift(
                int nr_polarizations,
                int gridsize,
                std::complex<float> *grid);
            void fftshift(
                int nr_polarizations,
                int gridsize,
                std::complex<float> *grid);

            void adder(
                const int jobsize,
                const void *_metadata,
                const void *_subgrid,
                      void *grid,
                const int gridsize,
                const int subgridsize,
                const int nr_polarizations);

            void splitter(
                const int jobsize,
                const void *metadata,
                      void *subgrid,
                const void *grid,
                const int gridsize,
                const int subgridsize,
                const int nr_polarizations);

        } // namespace knc
    } // namespace kernel
} // namespace idg

#pragma omp end declare target
#endif
