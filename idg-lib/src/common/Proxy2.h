#ifndef IDG_PROXY2_H_
#define IDG_PROXY2_H_

#include <complex>
#include <vector>
#include <limits>
#include <cstring>
#include <utility> // pair

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"
#include "CompileConstants.h"
#include "Types.h"
#include "Plan.h"

namespace idg {
    enum DomainAtoDomainB {
        FourierDomainToImageDomain,
        ImageDomainToFourierDomain
    };

    typedef std::string Compiler;
    typedef std::string Compilerflags;
}


namespace idg {
    namespace proxy {

        class Proxy2
        {
            public:
                Proxy2(
                    CompileConstants c
                ) : mConstants(c) {}

                virtual ~Proxy2() {}

                /*
                    High level routines
                */
                //! Grid the visibilities onto a uniform grid
                virtual void gridding(
                    const Plan& plan,
                    const float w_offset, // in lambda
                    const float cell_size, // TODO: unit?
                    const unsigned int kernel_size, // full width in pixels
                    const Array1D<float>& frequencies,
                    const Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) = 0;

                void gridding(
                    const float w_offset,
                    const float cell_size,
                    const unsigned int kernel_size,
                    const Array1D<float>& frequencies,
                    const Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal);

                virtual void gridding(
                    float w_offset,
                    const float cell_size,
                    unsigned int kernel_size,
                    float* frequencies,
                    unsigned int nr_channels,
                    std::complex<float>* visibilities,
                    unsigned int visibilities_nr_baselines,
                    unsigned int visibilities_nr_timesteps,
                    unsigned int visibilities_nr_channels,
                    unsigned int visibilities_nr_correlations,
                    float* uvw,
                    unsigned int uvw_nr_baselines,
                    unsigned int uvw_nr_timesteps,
                    unsigned int uvw_nr_coordinates, // 3 (u, v, w)
                    unsigned int* baselines,
                    unsigned int baselines_nr_baselines,
                    unsigned int baselines_two, // antenna1, antenna2
                    std::complex<float>* grid,
                    unsigned int grid_nr_correlations,
                    unsigned int grid_height,
                    unsigned int grid_width,
                    std::complex<float>* aterms,
                    unsigned int aterms_nr_timeslots,
                    unsigned int aterms_nr_stations,
                    unsigned int aterms_aterm_height,
                    unsigned int aterms_aterm_width,
                    unsigned int aterms_nr_correlations,
                    unsigned int* aterms_offsets,
                    unsigned int aterms_offsets_nr_timeslots_plus_one,
                    float* spheroidal,
                    unsigned int spheroidal_height,
                    unsigned int spheroidal_width);

                //! Degrid the visibilities from a uniform grid
                virtual void degridding(
                    const Plan& plan,
                    const float w_offset, // in lambda
                    const float cell_size, // TODO: unit?
                    const unsigned int kernel_size, // full width in pixels
                    const Array1D<float>& frequencies,
                    Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) = 0;

                void degridding(
                    const float w_offset,
                    const float cell_size,
                    const unsigned int kernel_size,
                    const Array1D<float>& frequencies,
                    Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal);

                void degridding(
                    float w_offset,
                    float cell_size,
                    unsigned int kernel_size,
                    float* frequencies,
                    unsigned int nr_channels,
                    std::complex<float>* visibilities,
                    unsigned int visibilities_nr_baselines,
                    unsigned int visibilities_nr_timesteps,
                    unsigned int visibilities_nr_channels,
                    unsigned int visibilities_nr_correlations,
                    float* uvw,
                    unsigned int uvw_nr_baselines,
                    unsigned int uvw_nr_timesteps,
                    unsigned int uvw_nr_coordinates, // 3 (u, v, w)
                    unsigned int* baselines,
                    unsigned int baselines_nr_baselines,
                    unsigned int baselines_two, // antenna1, antenna2
                    std::complex<float>* grid,
                    unsigned int grid_nr_correlations,
                    unsigned int grid_height,
                    unsigned int grid_width,
                    std::complex<float>* aterms,
                    unsigned int aterms_nr_timeslots,
                    unsigned int aterms_nr_stations,
                    unsigned int aterms_aterm_height,
                    unsigned int aterms_aterm_width,
                    unsigned int aterms_nr_correlations,
                    unsigned int* aterms_offsets,
                    unsigned int aterms_offsets_nr_timeslots_plus_one,
                    float* spheroidal,
                    unsigned int spheroidal_height,
                    unsigned int spheroidal_width);

                //! Applyies (inverse) Fourier transform to grid
                virtual void transform(
                    DomainAtoDomainB direction,
                    const Array3D<std::complex<float>>& grid) = 0;

                void transform(
                    DomainAtoDomainB direction,
                    std::complex<float>* grid,
                    unsigned int grid_nr_correlations,
                    unsigned int grid_height,
                    unsigned int grid_width);

                // Auxiliary: set and get methods
                unsigned int get_nr_correlations() const {
                    return mConstants.get_nr_correlations(); }
                unsigned int get_subgrid_size() const {
                    return mConstants.get_subgrid_size(); }

            protected:
                void check_dimensions(
                    unsigned int frequencies_nr_channels,
                    unsigned int visibilities_nr_baselines,
                    unsigned int visibilities_nr_timesteps,
                    unsigned int visibilities_nr_channels,
                    unsigned int visibilities_nr_correlations,
                    unsigned int uvw_nr_baselines,
                    unsigned int uvw_nr_timesteps,
                    unsigned int uvw_nr_coordinates,
                    unsigned int baselines_nr_baselines,
                    unsigned int baselines_two,
                    unsigned int grid_nr_correlations,
                    unsigned int grid_height,
                    unsigned int grid_width,
                    unsigned int aterms_nr_timeslots,
                    unsigned int aterms_nr_stations,
                    unsigned int aterms_aterm_height,
                    unsigned int aterms_aterm_width,
                    unsigned int aterms_nr_correlations,
                    unsigned int aterms_offsets_nr_timeslots_plus_one,
                    unsigned int spheroidal_height,
                    unsigned int spheroidal_width) const;

                void check_dimensions(
                    const Array1D<float>& frequencies,
                    const Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) const;

                Array1D<float> compute_wavenumbers(
                    const Array1D<float>& frequencies) const;

                CompileConstants mConstants;

        }; // end class Proxy2

    } // namespace proxy
} // namespace idg

#endif
