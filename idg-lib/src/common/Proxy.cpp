#include <ThrowAssert.hpp> // assert
#include <cmath> // M_PI

#include "Proxy.h"

namespace idg {
    namespace proxy {
        Proxy::Proxy() {
            grid_ptr = NULL;
        }

        Proxy::~Proxy() {
            if (grid_ptr != NULL) {
                delete grid_ptr;
            }
        }

        void Proxy::gridding(
            const Plan& plan,
            const float w_step, // in lambda
            const Array1D<float>& shift,
            const float cell_size, // TODO: unit?
            const unsigned int kernel_size, // full width in pixels
            const unsigned int subgrid_size,
            const Array1D<float>& frequencies,
            const Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            check_dimensions(
                subgrid_size, frequencies, visibilities, uvw, baselines,
                grid, aterms, aterms_offsets, spheroidal);

            if ((w_step != 0.0) && (!supports_wstack_gridding())) {
                throw std::invalid_argument("w_step is not zero, but this Proxy does not support gridding with W-stacking.");
            }
            do_gridding(plan, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities, uvw, baselines, grid, aterms, aterms_offsets, spheroidal);
        }

        void Proxy::gridding(
            const float w_step,
            const Array1D<float>& shift,
            const float cell_size,
            const unsigned int kernel_size,
            const unsigned int subgrid_size,
            const Array1D<float>& frequencies,
            const Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            auto grid_size        = grid.get_x_dim();
            auto nr_w_layers      = grid.get_w_dim();

            Plan::Options options;
            options.w_step = w_step;
            options.nr_w_layers = nr_w_layers;

            Plan plan(
                kernel_size,
                subgrid_size,
                grid_size,
                cell_size,
                frequencies,
                uvw,
                baselines,
                aterms_offsets,
                options);

            gridding(
                plan,
                w_step,
                shift,
                cell_size,
                kernel_size,
                subgrid_size,
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
            float* shift,
            float cell_size,
            unsigned int kernel_size,
            unsigned int subgrid_size,
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
                subgrid_size,
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

            Array1D<float> shift_(
                shift, 3);
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
                shift_,
                cell_size,
                kernel_size,
                subgrid_size,
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
            const Array1D<float>& shift,
            const float cell_size, // TODO: unit?
            unsigned int kernel_size, // full width in pixels
            unsigned int subgrid_size,
            const Array1D<float>& frequencies,
            Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            const Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            check_dimensions(
                subgrid_size, frequencies, visibilities, uvw, baselines,
                grid, aterms, aterms_offsets, spheroidal);

            if ((w_step != 0.0) && (!supports_wstack_degridding())) {
                throw std::invalid_argument("w_step is not zero, but this Proxy does not support degridding with W-stacking.");
            }
            do_degridding(plan, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities, uvw, baselines, grid, aterms, aterms_offsets, spheroidal);
        }

        void Proxy::degridding(
            const float w_step,
            const Array1D<float>& shift,
            const float cell_size,
            const unsigned int kernel_size,
            const unsigned int subgrid_size,
            const Array1D<float>& frequencies,
            Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            const Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            auto grid_size        = grid.get_x_dim();
            auto nr_w_layers      = grid.get_w_dim();

            Plan::Options options;
            options.w_step = w_step;
            options.nr_w_layers = nr_w_layers;

            Plan plan(
                kernel_size,
                subgrid_size,
                grid_size,
                cell_size,
                frequencies,
                uvw, baselines,
                aterms_offsets,
                options);

            degridding(
                plan,
                w_step,
                shift,
                cell_size,
                kernel_size,
                subgrid_size,
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
            float* shift,
            float cell_size,
            unsigned int kernel_size,
            unsigned int subgrid_size,
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
                subgrid_size,
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

            Array1D<float> shift_(
                shift, 3);
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
                shift_,
                cell_size,
                kernel_size,
                subgrid_size,
                frequencies_,
                visibilities_,
                uvw_,
                baselines_,
                grid_,
                aterms_,
                aterms_offsets_,
                spheroidal_);
        }

        void Proxy::set_avg_aterm_correction(
            const Array4D<std::complex<float>>& avg_aterm_correction)
        {
            if (!supports_avg_aterm_correction())
            {
                throw exception::NotImplemented("This proxy does not support average aterm correction");
            }

//             check_dimensions_avg_aterm_correction();
            std::complex<float> *data = avg_aterm_correction.data();
            size_t size = avg_aterm_correction.get_x_dim() *
                            avg_aterm_correction.get_y_dim() *
                            avg_aterm_correction.get_z_dim() *
                            avg_aterm_correction.get_w_dim();
            m_avg_aterm_correction.resize(size);
            std::copy(data, data+size, m_avg_aterm_correction.begin());
        }

        void Proxy::unset_avg_aterm_correction()
        {
            m_avg_aterm_correction.resize(0);
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
            throw_assert(grid_height == grid_width, ""); // TODO: remove restriction
            throw_assert(grid_nr_correlations == 1 || grid_nr_correlations == 4, "");

            Array3D<std::complex<float>> grid_(
                grid, grid_nr_correlations, grid_height, grid_width);

            transform(direction, grid_);
        }

        void Proxy::check_dimensions(
            unsigned int subgrid_size,
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
            throw_assert(frequencies_nr_channels > 0, "");
            throw_assert(frequencies_nr_channels == visibilities_nr_channels, "");
            throw_assert(visibilities_nr_baselines == uvw_nr_baselines, "");
            throw_assert(visibilities_nr_baselines == baselines_nr_baselines, "");
            throw_assert(visibilities_nr_timesteps == uvw_nr_timesteps, "");
            throw_assert(visibilities_nr_correlations == 1 || visibilities_nr_correlations == 4, "");
            throw_assert(visibilities_nr_correlations == grid_nr_correlations, "");
            throw_assert(visibilities_nr_correlations == aterms_nr_correlations, "");
            throw_assert(uvw_nr_coordinates == 3, "");
            throw_assert(baselines_two == 2, "");
            throw_assert(grid_height == grid_width, ""); // TODO: remove restriction
            throw_assert(aterms_nr_timeslots + 1 == aterms_offsets_nr_timeslots_plus_one, "");
            throw_assert(aterms_aterm_height == aterms_aterm_width, ""); // TODO: remove restriction
            throw_assert(spheroidal_height == subgrid_size, "");
            throw_assert(spheroidal_height == subgrid_size, "");
        }

        void Proxy::check_dimensions(
            unsigned int subgrid_size,
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
                subgrid_size,
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

         Grid Proxy::get_grid(
            size_t nr_w_layers,
            size_t nr_correlations,
            size_t height,
            size_t width)
        {
            if (grid_ptr != NULL) {
                delete grid_ptr;
            }
            grid_ptr = new std::complex<float>[nr_w_layers*nr_correlations*height*width];
            Grid grid(grid_ptr, nr_w_layers, nr_correlations, height, width);
            grid.zero();
            return grid;
        }

        void Proxy::free_grid(
            Grid& grid)
        {
            assert(grid_ptr == grid.data());
            delete grid_ptr;
            grid_ptr = NULL;
        }

        void Proxy::run_gridding(
            const Plan& plan,
            const float w_step,
            const Array1D<float>& shift,
            const float cell_size,
            const unsigned int kernel_size,
            const unsigned int subgrid_size,
            const Array1D<float>& frequencies,
            const Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            gridding(
                plan,
                w_step,
                shift,
                cell_size,
                kernel_size,
                subgrid_size,
                frequencies,
                visibilities,
                uvw,
                baselines,
                grid,
                aterms,
                aterms_offsets,
                spheroidal);
        }

        void Proxy::run_degridding(
            const Plan& plan,
            const float w_step,
            const Array1D<float>& shift,
            const float cell_size,
            const unsigned int kernel_size,
            const unsigned int subgrid_size,
            const Array1D<float>& frequencies,
            Array3D<Visibility<std::complex<float>>>& visibilities,
            const Array2D<UVWCoordinate<float>>& uvw,
            const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
            const Grid& grid,
            const Array4D<Matrix2x2<std::complex<float>>>& aterms,
            const Array1D<unsigned int>& aterms_offsets,
            const Array2D<float>& spheroidal)
        {
            degridding(
                plan,
                w_step,
                shift,
                cell_size,
                kernel_size,
                subgrid_size,
                frequencies,
                visibilities,
                uvw,
                baselines,
                grid,
                aterms,
                aterms_offsets,
                spheroidal);
        }


    } // end namespace proxy
} // end namespace idg

#include "ProxyC.h"
