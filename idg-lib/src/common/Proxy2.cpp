#include <cassert> // assert

#include "Proxy2.h"

using namespace std;

namespace idg {
    namespace proxy {
        void Proxy2::gridding(
            float w_offset,
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
            unsigned int uvw_nr_coordinates,
            unsigned int* baselines,
            unsigned int baselines_nr_baselines,
            unsigned int baselines_two,
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
            unsigned int spheroidal_width)
        {
            (kernel_size > 0);
            assert(nr_channels > 0);
            assert(nr_channels == visibilities_nr_channels);
            assert(visibilities_nr_baselines == uvw_nr_baselines);
            assert(visibilities_nr_baselines == baselines_nr_baselines);
            assert(visibilities_nr_timesteps == uvw_nr_timesteps);
            assert(visibilities_nr_correlations == grid_nr_correlations);
            assert(visibilities_nr_correlations == aterms_nr_correlations);
            assert(uvw_nr_coordinates == 3);
            assert(baselines_two == 2);
            assert(grid_height == grid_width); // TODO: remove restriction
            assert(aterms_nr_timeslots + 1 == aterms_offsets_nr_timeslots_plus_one);
            assert(aterms_aterm_height == aterms_aterm_width); // TODO: remove restriction
            assert(spheroidal_height == spheroidal_width); // TODO: remove restriction

            Array1D<float> frequencies_(
                frequencies, nr_channels);
            Array3D<Visibility<std::complex<float>>> visibilities_(
                (Visibility<std::complex<float>> *) visibilities, visibilities_nr_baselines,
                visibilities_nr_timesteps, visibilities_nr_channels);
            Array2D<UVWCoordinate<float>> uvw_(
                (UVWCoordinate<float> *) uvw, uvw_nr_baselines, uvw_nr_timesteps);
            Array1D<std::pair<unsigned int,unsigned int>> baselines_(
                (std::pair<unsigned int,unsigned int> *) baselines, baselines_nr_baselines);
            Array3D<std::complex<float>> grid_(
                grid, grid_nr_correlations, grid_height, grid_width);
            Array4D<Matrix2x2<std::complex<float>>> aterms_(
                (Matrix2x2<std::complex<float>> *) aterms, aterms_nr_timeslots, aterms_nr_stations,
                aterms_aterm_height, aterms_aterm_width);
            Array1D<unsigned int> aterms_offsets_(
                aterms_offsets, aterms_offsets_nr_timeslots_plus_one);
            Array2D<float> spheroidal_(
                spheroidal, spheroidal_height, spheroidal_width);

            gridding(
                w_offset,
                kernel_size,
                frequencies_,
                visibilities_,
                uvw_,
                baselines_,
                grid_,
                aterms_,
                aterms_offsets_,
                spheroidal_);
        }


        void Proxy2::degridding(
            float w_offset,
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
            unsigned int uvw_nr_coordinates,
            unsigned int* baselines,
            unsigned int baselines_nr_baselines,
            unsigned int baselines_two,
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
            unsigned int spheroidal_width)
        {
            (kernel_size > 0);
            assert(nr_channels > 0);
            assert(nr_channels == visibilities_nr_channels);
            assert(visibilities_nr_baselines == uvw_nr_baselines);
            assert(visibilities_nr_baselines == baselines_nr_baselines);
            assert(visibilities_nr_timesteps == uvw_nr_timesteps);
            assert(visibilities_nr_correlations == grid_nr_correlations);
            assert(visibilities_nr_correlations == aterms_nr_correlations);
            assert(uvw_nr_coordinates == 3);
            assert(baselines_two == 2);
            assert(grid_height == grid_width); // TODO: remove restriction
            assert(aterms_nr_timeslots + 1 == aterms_offsets_nr_timeslots_plus_one);
            assert(aterms_aterm_height == aterms_aterm_width); // TODO: remove restriction
            assert(spheroidal_height == spheroidal_width); // TODO: remove restriction

            Array1D<float> frequencies_(
                frequencies, nr_channels);
            Array3D<Visibility<std::complex<float>>> visibilities_(
                (Visibility<std::complex<float>> *) visibilities, visibilities_nr_baselines,
                visibilities_nr_timesteps, visibilities_nr_channels);
            Array2D<UVWCoordinate<float>> uvw_(
                (UVWCoordinate<float> *) uvw, uvw_nr_baselines, uvw_nr_timesteps);
            Array1D<std::pair<unsigned int,unsigned int>> baselines_(
                (std::pair<unsigned int,unsigned int> *) baselines, baselines_nr_baselines);
            Array3D<std::complex<float>> grid_(
                grid, grid_nr_correlations, grid_height, grid_width);
            Array4D<Matrix2x2<std::complex<float>>> aterms_(
                (Matrix2x2<std::complex<float>> *) aterms, aterms_nr_timeslots, aterms_nr_stations,
                aterms_aterm_height, aterms_aterm_width);
            Array1D<unsigned int> aterms_offsets_(
                aterms_offsets, aterms_offsets_nr_timeslots_plus_one);
            Array2D<float> spheroidal_(
                spheroidal, spheroidal_height, spheroidal_width);

            degridding(
                w_offset,
                kernel_size,
                frequencies_,
                visibilities_,
                uvw_,
                baselines_,
                grid_,
                aterms_,
                aterms_offsets_,
                spheroidal_);
        }


    } // end namespace proxy
} // end namespace idg

// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::Proxy2 Proxy2_t;

    int Proxy2_get_nr_correlations(Proxy2_t* p)
    {
        return p->get_nr_correlations();
    }

    int Proxy2_get_subgrid_size(Proxy2_t* p)
    {
        return p->get_subgrid_size();
    }
} // end extern "C"
