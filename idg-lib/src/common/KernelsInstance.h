#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <cmath>
#include <cassert>

#include "idg-common.h"


namespace idg {
    namespace kernel {

        class KernelsInstance
        {
            public:
                KernelsInstance(
                    CompileConstants& constants) :
                    mConstants(constants) {}

                /*
                    Flops/bytes computation
                */
                uint64_t flops_gridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids) const;
                uint64_t bytes_gridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids) const;
                uint64_t flops_degridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids) const;
                uint64_t bytes_degridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids) const;
                uint64_t flops_fft(
                    uint64_t size,
                    uint64_t batch) const;
                uint64_t bytes_fft(
                    uint64_t size,
                    uint64_t batch) const;
                uint64_t flops_adder(
                    uint64_t nr_subgrids) const;
                uint64_t bytes_adder(
                    uint64_t nr_subgrids) const;
                uint64_t flops_splitter(
                    uint64_t nr_subgrids) const;
                uint64_t bytes_splitter(
                    uint64_t nr_subgrids) const;
                uint64_t flops_scaler(
                    uint64_t nr_subgrids) const;
                uint64_t bytes_scaler(
                    uint64_t nr_subgrids) const;

                /*
                    Misc math routines
                */
                void shift(
                    Array3D<std::complex<float>>& data) const;

                void scale(
                    Array3D<std::complex<float>>& data,
                    std::complex<float> scale) const;

                /*
                    Sizeof routines
                */
                uint64_t sizeof_visibilities(
                    unsigned int nr_baselines,
                    unsigned int nr_timesteps,
                    unsigned int nr_channels) const;

                uint64_t sizeof_uvw(
                    unsigned int nr_baselines,
                    unsigned int nr_timesteps) const;

                uint64_t sizeof_subgrids(
                    unsigned int nr_subgrids,
                    unsigned int subgrid_size) const;

                uint64_t sizeof_metadata(
                    unsigned int nr_subgrids) const;

                uint64_t sizeof_grid(
                    unsigned int grid_size) const;

                uint64_t sizeof_wavenumbers(
                    unsigned int nr_channels) const;

                uint64_t sizeof_aterms(
                    unsigned int nr_stations,
                    unsigned int nr_timeslots,
                    unsigned int subgrid_size) const;

                uint64_t sizeof_spheroidal(
                    unsigned int subgrid_size) const;

            protected:
                CompileConstants mConstants;

        }; // end class KernelsInstance

    } // namespace kernel
} // namespace idg
#endif
