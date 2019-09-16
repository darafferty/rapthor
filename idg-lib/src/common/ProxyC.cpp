#include "Proxy.h"
#include "ProxyC.h"


extern "C" {

    void Proxy_gridding(
        Proxy* p,
        float w_step,
        float* shift,
        const float cell_size,
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
        reinterpret_cast<idg::proxy::Proxy*>(p)->gridding(
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
            frequencies,
            nr_channels,
            visibilities,
            visibilities_nr_baselines,
            visibilities_nr_timesteps,
            visibilities_nr_channels,
            visibilities_nr_correlations,
            uvw,
            uvw_nr_baselines,
            uvw_nr_timesteps,
            uvw_nr_coordinates,
            baselines,
            baselines_nr_baselines,
            baselines_two,
            grid,
            grid_nr_correlations,
            grid_height,
            grid_width,
            aterms,
            aterms_nr_timeslots,
            aterms_nr_stations,
            aterms_aterm_height,
            aterms_aterm_width,
            aterms_nr_correlations,
            aterms_offsets,
            aterms_offsets_nr_timeslots_plus_one,
            spheroidal,
            spheroidal_height,
            spheroidal_width
        );
    }

     void Proxy_degridding(
        Proxy* p,
        float w_step,
        float* shift,
        const float cell_size,
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
        reinterpret_cast<idg::proxy::Proxy*>(p)->degridding(
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
            frequencies,
            nr_channels,
            visibilities,
            visibilities_nr_baselines,
            visibilities_nr_timesteps,
            visibilities_nr_channels,
            visibilities_nr_correlations,
            uvw,
            uvw_nr_baselines,
            uvw_nr_timesteps,
            uvw_nr_coordinates,
            baselines,
            baselines_nr_baselines,
            baselines_two,
            grid,
            grid_nr_correlations,
            grid_height,
            grid_width,
            aterms,
            aterms_nr_timeslots,
            aterms_nr_stations,
            aterms_aterm_height,
            aterms_aterm_width,
            aterms_nr_correlations,
            aterms_offsets,
            aterms_offsets_nr_timeslots_plus_one,
            spheroidal,
            spheroidal_height,
            spheroidal_width
        );
    }

     void Proxy_calibrate_init(
        Proxy* p,
        float w_step,
        float* shift,
        const float cell_size,
        unsigned int kernel_size,
        unsigned int subgrid_size,
        unsigned int nr_channels,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_timeslots,
        unsigned int nr_correlations,
        unsigned int grid_height,
        unsigned int grid_width,
        float* frequencies,
        std::complex<float>* visibilities,
        float* weights,
        float* uvw,
        unsigned int* baselines,
        std::complex<float>* grid,
        unsigned int* aterms_offsets,
        float* spheroidal)
    {
        idg::Array1D<float> shift_(
            shift, 3);
        idg::Array1D<float> frequencies_(
            frequencies, nr_channels);
        idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_(
            (idg::Visibility<std::complex<float>> *) visibilities, nr_baselines,
            nr_timesteps, nr_channels);
        idg::Array3D<idg::Visibility<float>> weights_(
            (idg::Visibility<float> *) weights, nr_baselines,
            nr_timesteps, nr_channels);
        idg::Array2D<idg::UVW<float>> uvw_(
            (idg::UVW<float> *) uvw, nr_baselines, nr_timesteps);
        idg::Array1D<std::pair<unsigned int,unsigned int>> baselines_(
            (std::pair<unsigned int,unsigned int> *) baselines, nr_baselines);
        idg::Grid grid_(
            grid, 1, nr_correlations, grid_height, grid_width);
        idg::Array1D<unsigned int> aterms_offsets_(aterms_offsets, nr_timeslots+1);
        idg::Array2D<float> spheroidal_(
            spheroidal, subgrid_size, subgrid_size);

        reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_init(
            w_step,
            shift_,
            cell_size,
            kernel_size,
            subgrid_size,
            frequencies_,
            visibilities_,
            weights_,
            uvw_,
            baselines_,
            grid_,
            aterms_offsets_,
            spheroidal_);
    }

    void Proxy_calibrate_update(
        Proxy* p,
        const unsigned int antenna_nr,
        const unsigned int subgrid_size,
        const unsigned int nr_antennas,
        const unsigned int nr_timeslots,
        const unsigned int nr_terms,
        std::complex<float>* aterms,
        std::complex<float>* aterm_derivatives,
        double* hessian,
        double* gradient,
        double *residual)
    {
        idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms), nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
        idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives), nr_timeslots, nr_terms, subgrid_size, subgrid_size);
        idg::Array3D<double> hessian_(hessian, nr_timeslots, nr_terms, nr_terms);
        idg::Array2D<double> gradient_(gradient, nr_timeslots, nr_terms);
        reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_update(antenna_nr, aterms_, aterm_derivatives_, hessian_, gradient_, *residual);
    }

    void Proxy_calibrate_finish(
        Proxy* p)
    {
        reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_finish();
    }

    void Proxy_calibrate_init_hessian_vector_product(Proxy* p)
    {
        reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_init_hessian_vector_product();
    }

    void Proxy_calibrate_hessian_vector_product1(
        Proxy* p,
        const unsigned int antenna_nr,
        const unsigned int subgrid_size,
        const unsigned int nr_antennas,
        const unsigned int nr_timeslots,
        const unsigned int nr_terms,
        std::complex<float>* aterms,
        std::complex<float>* aterm_derivatives,
        float* parameter_vector)
    {
        idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms), nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
        idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives), nr_timeslots, nr_terms, subgrid_size, subgrid_size);
        idg::Array2D<float> parameter_vector_(parameter_vector, nr_timeslots, nr_terms);
        reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_update_hessian_vector_product1(antenna_nr, aterms_, aterm_derivatives_, parameter_vector_);
    }

    void Proxy_calibrate_hessian_vector_product2(
        Proxy* p,
        const unsigned int antenna_nr,
        const unsigned int subgrid_size,
        const unsigned int nr_antennas,
        const unsigned int nr_timeslots,
        const unsigned int nr_terms,
        std::complex<float>* aterms,
        std::complex<float>* aterm_derivatives,
        float* parameter_vector)
    {
        idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms), nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
        idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives), nr_timeslots, nr_terms, subgrid_size, subgrid_size);
        idg::Array2D<float> parameter_vector_(parameter_vector, nr_timeslots, nr_terms);
        reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_update_hessian_vector_product2(antenna_nr, aterms_, aterm_derivatives_, parameter_vector_);
    }

    void Proxy_transform(
        Proxy* p,
        int direction,
        std::complex<float>* grid,
        unsigned int grid_nr_correlations,
        unsigned int grid_height,
        unsigned int grid_width)
    {
        if (direction!=0) {
            reinterpret_cast<idg::proxy::Proxy*>(p)->transform(
                idg::ImageDomainToFourierDomain,
                grid,
                grid_nr_correlations,
                grid_height,
                grid_width);
       } else {
            reinterpret_cast<idg::proxy::Proxy*>(p)->transform(idg::FourierDomainToImageDomain,
                 grid,
                 grid_nr_correlations,
                 grid_height,
                 grid_width);
        }
    }

    void Proxy_destroy(Proxy* p) {
       delete reinterpret_cast<idg::proxy::Proxy*>(p);
    }

    void* Proxy_get_grid(
        Proxy* p,
        unsigned int nr_correlations,
        unsigned int grid_size)
    {
        const unsigned int nr_w_layers = 1;
        idg::Grid grid = reinterpret_cast<idg::proxy::Proxy*>(p)->get_grid(nr_w_layers, nr_correlations, grid_size, grid_size);
        return grid.data();
    }
}
