#include <cassert> // assert
#include <cmath> // M_PI

#include "Proxy.h"

namespace idg {
    namespace proxy {
        
        void Proxy::gridding(
            const Plan& plan,
            const float w_step, // in lambda
            const float cell_size, // TODO: unit?
            const unsigned int kernel_size, // full width in pixels
            const Array1D<float>& frequencies,
            const Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            if ((w_step != 0.0) && (!supports_wstack_gridding())) {
                throw std::invalid_argument("w_step is not zero, but this Proxy does not support gridding with W-stacking.");
            }
            do_gridding(plan, w_step, cell_size, kernel_size, frequencies, visibilities, uvw, baselines, grid, aterms, aterms_offsets, spheroidal);
        }
        
        void Proxy::gridding(
            const float w_step,
            const float cell_size,
            const unsigned int kernel_size,
            const Array1D<float>& frequencies,
            const Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            auto subgrid_size     = mConstants.get_subgrid_size();
            auto nr_polarizations = mConstants.get_nr_correlations();
            auto grid_size        = grid.get_x_dim();

            Plan plan(
                kernel_size,
                subgrid_size,
                grid_size,
                cell_size,
                frequencies,
                uvw,
                baselines,
                aterms_offsets);

            gridding(
                plan,
                w_step,
                cell_size,
                kernel_size,
                frequencies,
                visibilities,
                uvw,
                baselines,
                grid,
                aterms,
                aterms_offsets,
                spheroidal);
        }

        void Proxy::gridding(
            float w_step,
            float cell_size,
            unsigned int kernel_size,
            float* frequencies,
            unsigned int frequencies_nr_channels,
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
            check_dimensions(
                frequencies_nr_channels,
                visibilities_nr_baselines,
                visibilities_nr_timesteps,
                visibilities_nr_channels,
                visibilities_nr_correlations,
                uvw_nr_baselines,
                uvw_nr_timesteps,
                uvw_nr_coordinates,
                baselines_nr_baselines,
                baselines_two,
                grid_nr_correlations,
                grid_height,
                grid_width,
                aterms_nr_timeslots,
                aterms_nr_stations,
                aterms_aterm_height,
                aterms_aterm_width,
                aterms_nr_correlations,
                aterms_offsets_nr_timeslots_plus_one,
                spheroidal_height,
                spheroidal_width);

            Array1D<float> frequencies_(
                frequencies, frequencies_nr_channels);
            Array3D<Visibility<std::complex<float>>> visibilities_(
                (Visibility<std::complex<float>> *) visibilities, visibilities_nr_baselines,
                visibilities_nr_timesteps, visibilities_nr_channels);
            Array2D<UVWCoordinate<float>> uvw_(
                (UVWCoordinate<float> *) uvw, uvw_nr_baselines, uvw_nr_timesteps);
            Array1D<std::pair<unsigned int,unsigned int>> baselines_(
                (std::pair<unsigned int,unsigned int> *) baselines, baselines_nr_baselines);
            Grid grid_(
                grid, 1, grid_nr_correlations, grid_height, grid_width);
            Array4D<Matrix2x2<std::complex<float>>> aterms_(
                (Matrix2x2<std::complex<float>> *) aterms, aterms_nr_timeslots, aterms_nr_stations,
                aterms_aterm_height, aterms_aterm_width);
            Array1D<unsigned int> aterms_offsets_(
                aterms_offsets, aterms_offsets_nr_timeslots_plus_one);
            Array2D<float> spheroidal_(
                spheroidal, spheroidal_height, spheroidal_width);

            gridding(
                w_step,
                cell_size,
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


        void Proxy::degridding(
            const Plan& plan,
            const float w_step, // in lambda
            const float cell_size, // TODO: unit?
            const unsigned int kernel_size, // full width in pixels
            const Array1D<float>& frequencies,
            Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            const Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            if ((w_step != 0.0) && (!supports_wstack_degridding())) {
                throw std::invalid_argument("w_step is not zero, but this Proxy does not support degridding with W-stacking.");
            }
            do_degridding(plan, w_step, cell_size, kernel_size, frequencies, visibilities, uvw, baselines, grid, aterms, aterms_offsets, spheroidal);
        }
        

        void Proxy::degridding(
            const float w_step,
            const float cell_size,
            const unsigned int kernel_size,
            const Array1D<float>& frequencies,
            Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            const Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            auto subgrid_size     = mConstants.get_subgrid_size();
            auto nr_polarizations = mConstants.get_nr_correlations();
            auto grid_size        = grid.get_x_dim();

            Plan plan(
                kernel_size, subgrid_size, grid_size, cell_size,
                frequencies, uvw, baselines, aterms_offsets, w_step, 
                grid.get_nr_w_layers()-1);

            degridding(
                plan,
                w_step,
                cell_size,
                kernel_size,
                frequencies,
                visibilities,
                uvw,
                baselines,
                grid,
                aterms,
                aterms_offsets,
                spheroidal);
        }


        void Proxy::degridding(
            float w_step,
            float cell_size,
            unsigned int kernel_size,
            float* frequencies,
            unsigned int frequencies_nr_channels,
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
            check_dimensions(
                frequencies_nr_channels,
                visibilities_nr_baselines,
                visibilities_nr_timesteps,
                visibilities_nr_channels,
                visibilities_nr_correlations,
                uvw_nr_baselines,
                uvw_nr_timesteps,
                uvw_nr_coordinates,
                baselines_nr_baselines,
                baselines_two,
                grid_nr_correlations,
                grid_height,
                grid_width,
                aterms_nr_timeslots,
                aterms_nr_stations,
                aterms_aterm_height,
                aterms_aterm_width,
                aterms_nr_correlations,
                aterms_offsets_nr_timeslots_plus_one,
                spheroidal_height,
                spheroidal_width);

            Array1D<float> frequencies_(
                frequencies, frequencies_nr_channels);
            Array3D<Visibility<std::complex<float>>> visibilities_(
                (Visibility<std::complex<float>> *) visibilities, visibilities_nr_baselines,
                visibilities_nr_timesteps, visibilities_nr_channels);
            Array2D<UVWCoordinate<float>> uvw_(
                (UVWCoordinate<float> *) uvw, uvw_nr_baselines, uvw_nr_timesteps);
            Array1D<std::pair<unsigned int,unsigned int>> baselines_(
                (std::pair<unsigned int,unsigned int> *) baselines, baselines_nr_baselines);
            Grid grid_(
                grid, 1, grid_nr_correlations, grid_height, grid_width);
            Array4D<Matrix2x2<std::complex<float>>> aterms_(
                (Matrix2x2<std::complex<float>> *) aterms, aterms_nr_timeslots, aterms_nr_stations,
                aterms_aterm_height, aterms_aterm_width);
            Array1D<unsigned int> aterms_offsets_(
                aterms_offsets, aterms_offsets_nr_timeslots_plus_one);
            Array2D<float> spheroidal_(
                spheroidal, spheroidal_height, spheroidal_width);

            degridding(
                w_step,
                cell_size,
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
        
        
        void Proxy::transform(
            DomainAtoDomainB direction, 
            Array3D<std::complex<float>>& grid)
        {
            do_transform(direction, grid);
        }
        
        void Proxy::transform(
            DomainAtoDomainB direction,
            std::complex<float>* grid,
            unsigned int grid_nr_correlations,
            unsigned int grid_height,
            unsigned int grid_width)
        {
            assert(grid_height == grid_width); // TODO: remove restriction
            assert(grid_nr_correlations == 1 || grid_nr_correlations == 4);

            Array3D<std::complex<float>> grid_(
                grid, grid_nr_correlations, grid_height, grid_width);

            transform(direction, grid_);
        }

        void Proxy::check_dimensions(
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
            unsigned int spheroidal_width) const
        {
            assert(frequencies_nr_channels > 0);
            assert(frequencies_nr_channels == visibilities_nr_channels);
            assert(visibilities_nr_baselines == uvw_nr_baselines);
            assert(visibilities_nr_baselines == baselines_nr_baselines);
            assert(visibilities_nr_timesteps == uvw_nr_timesteps);
            assert(visibilities_nr_correlations == 1 || visibilities_nr_correlations == 4);
            assert(visibilities_nr_correlations == grid_nr_correlations);
            assert(visibilities_nr_correlations == aterms_nr_correlations);
            assert(uvw_nr_coordinates == 3);
            assert(baselines_two == 2);
            assert(grid_height == grid_width); // TODO: remove restriction
            assert(aterms_nr_timeslots + 1 == aterms_offsets_nr_timeslots_plus_one);
            assert(aterms_aterm_height == aterms_aterm_width); // TODO: remove restriction
            assert(spheroidal_height == spheroidal_width); // TODO: remove restriction
        }


        void Proxy::check_dimensions(
            const Array1D<float>& frequencies,
            const Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            const Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal) const
        {
            check_dimensions(
                frequencies.get_x_dim(),
                visibilities.get_z_dim(),
                visibilities.get_y_dim(),
                visibilities.get_x_dim(),
                4,
                uvw.get_y_dim(),
                uvw.get_x_dim(),
                3,
                baselines.get_x_dim(),
                2,
                grid.get_z_dim(),
                grid.get_y_dim(),
                grid.get_x_dim(),
                aterms.get_w_dim(),
                aterms.get_z_dim(),
                aterms.get_y_dim(),
                aterms.get_x_dim(),
                4,
                aterms_offsets.get_x_dim(),
                spheroidal.get_y_dim(),
                spheroidal.get_x_dim());
        }

        Array1D<float> Proxy::compute_wavenumbers(
            const Array1D<float>& frequencies) const
        {
            int nr_channels = frequencies.get_x_dim();
            Array1D<float> wavenumbers(nr_channels);

            const double speed_of_light = 299792458.0;
            for (int i = 0; i < nr_channels; i++) {
                wavenumbers(i) =  2 * M_PI * frequencies(i) / speed_of_light;
            }

            return wavenumbers;
        }

    } // end namespace proxy
} // end namespace idg

// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::Proxy Proxy_t;

    int Proxy_get_nr_correlations(Proxy_t* p)
    {
        return p->get_nr_correlations();
    }

    int Proxy_get_subgrid_size(Proxy_t* p)
    {
        return p->get_subgrid_size();
    }
} // end extern "C"
