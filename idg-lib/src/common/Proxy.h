#ifndef IDG_PROXY2_H_
#define IDG_PROXY2_H_

#include <complex>
#include <vector>
#include <limits>
#include <cstring>
#include <utility> // pair

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"
#include "Types.h"
#include "Plan.h"
#include "Report.h"
#include "Exception.h"


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

        class Proxy
        {
            public:
                Proxy();
                virtual ~Proxy();

                /*
                    High level routines
                */
                //! Grid the visibilities onto a uniform grid
                void gridding(
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
                    const Array2D<float>& spheroidal);

                void gridding(
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
                    const Array2D<float>& spheroidal);

                void gridding(
                    float w_step,
                    float* shift,
                    float cell_size,
                    unsigned int kernel_size,
                    unsigned int subgrid_size,
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

                void degridding(
                    const Plan& plan,
                    const float w_step, // in lambda
                    const Array1D<float>& shift,
                    const float cell_size, // TODO: unit?
                    const unsigned int kernel_size, // full width in pixels
                    const unsigned int subgrid_size,
                    const Array1D<float>& frequencies,
                    Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Grid& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal);

                void degridding(
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
                    const Array2D<float>& spheroidal);

                void degridding(
                    float w_step,
                    float* shift,
                    float cell_size,
                    unsigned int kernel_size,
                    unsigned int subgrid_size,
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
                void transform(
                    DomainAtoDomainB direction,
                    Array3D<std::complex<float>>& grid);

                void transform(
                    DomainAtoDomainB direction,
                    std::complex<float>* grid,
                    unsigned int grid_nr_correlations,
                    unsigned int grid_height,
                    unsigned int grid_width);

                //! Methods for W-stacking
                bool supports_wstack() {return (supports_wstack_gridding() && supports_wstack_degridding());}
                virtual bool supports_wstack_gridding() {return false;}
                virtual bool supports_wstack_degridding() {return false;}
                virtual bool supports_avg_aterm_correction() {return false;}

                void set_avg_aterm_correction(const Array4D<std::complex<float>>& avg_aterm_correction);
                void unset_avg_aterm_correction();

                //! Methods for memory management
                virtual Grid get_grid(
                    size_t nr_w_layers,
                    size_t nr_correlations,
                    size_t height,
                    size_t width);
                virtual void free_grid(
                    Grid& grid);


                /*
                    High level asynchronous routines
                */
                //! Dummy method, does nothing
                virtual void initialize(
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
                    const Grid& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) {};

                //! Dummy method, calls gridding()
                virtual void run_gridding(
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
                    const Array2D<float>& spheroidal);

                //! Dummy method, calls degridding()
                virtual void run_degridding(
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
                    const Array2D<float>& spheroidal);

                //! Dummy method
                virtual void finish_gridding() {};

                //! Dummy method
                virtual void finish_degridding() {};

                /*
                 * Misc
                 */
                inline void* allocate_memory(size_t size) { return malloc(size); }

            private:
                //! Degrid the visibilities from a uniform grid
                virtual void do_gridding(
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
                    const Array2D<float>& spheroidal) = 0;

                virtual void do_degridding(
                    const Plan& plan,
                    const float w_step, // in lambda
                    const Array1D<float>& shift,
                    const float cell_size, // TODO: unit?
                    const unsigned int kernel_size, // full width in pixels
                    const unsigned int subgrid_size,
                    const Array1D<float>& frequencies,
                    Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Grid& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) = 0;

                //! Applyies (inverse) Fourier transform to grid
                virtual void do_transform(
                    DomainAtoDomainB direction,
                    Array3D<std::complex<float>>& grid) = 0;

            protected:
                void check_dimensions(
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
                    unsigned int spheroidal_width) const;

                void check_dimensions(
                    unsigned int subgrid_size,
                    const Array1D<float>& frequencies,
                    const Array3D<Visibility<std::complex<float>>>& visibilities,
                    const Array2D<UVWCoordinate<float>>& uvw,
                    const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                    const Grid& grid,
                    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                    const Array1D<unsigned int>& aterms_offsets,
                    const Array2D<float>& spheroidal) const;

                Array1D<float> compute_wavenumbers(
                    const Array1D<float>& frequencies) const;

                const unsigned int nr_polarizations = 4;

                std::vector<std::complex<float>> m_avg_aterm_correction;

            private:
                std::complex<float>* grid_ptr = NULL;

            protected:
                Report report;

        }; // end class Proxy

    } // namespace proxy
} // namespace idg

#endif
