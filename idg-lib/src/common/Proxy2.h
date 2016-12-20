#ifndef IDG_PROXY2_H_
#define IDG_PROXY2_H_

#include <complex>
#include <vector>
#include <limits>
#include <cstring>
#include <utility> // pair

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"  // to be use in derived class
#include "Parameters2.h" // to be use in derived class
#include "Types.h"
#include "Plan.h"

namespace idg {
    enum DomainAtoDomainB {
        FourierDomainToImageDomain,
        ImageDomainToFourierDomain
    };

    /// typedefs
    typedef std::string Compiler;
    typedef std::string Compilerflags;
}


namespace idg {
    namespace proxy {

        class Proxy2
        {
            public:
                virtual ~Proxy2() {};

                /*
                    High level routines
                */
                //! Grid the visibilities onto a uniform grid
                virtual void gridding(
                    const float w_offset, // in lambda
                    const unsigned int kernel_size, // full width in pixels
                    const Array1D<float>& frequencies, // TODO: convert from wavenumbers
                    const Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) = 0;

                void gridding(
                    float w_offset, // in lambda
                    unsigned int kernel_size, // full width in pixels
                    float* frequencies, // TODO: convert from wavenumbers
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
                    const float w_offset, // in lambda
                    const unsigned int kernel_size, // full width in pixels
                    const Array1D<float>& frequencies, // TODO: convert from wavenumbers
                    Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Array3D<std::complex<float>>& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) = 0;

                void degridding(
                    float w_offset, // in lambda
                    unsigned int kernel_size, // full width in pixels
                    float* frequencies, // TODO: convert from wavenumbers
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
                    std::complex<float>* grid) = 0;

                // Auxiliary: set and get methods
                unsigned int get_nr_correlations() const {
                    return mParams.get_nr_correlations(); }
                unsigned int get_subgrid_size() const {
                    return mParams.get_subgrid_size(); }

            protected:
                CompileConstants mParams;
        };

    } // namespace proxy
} // namespace idg

#endif
