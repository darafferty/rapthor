extern "C" {

    struct Proxy;

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
        unsigned int spheroidal_width);

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
        unsigned int spheroidal_width);

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
        float* spheroidal);

     void Proxy_calibrate_update(
        Proxy* p,
        const unsigned int station_nr,
        const unsigned int subgrid_size,
        const unsigned int nr_stations,
        const unsigned int nr_time_slots,
        const unsigned int nr_terms,
        std::complex<float>* aterms,
        std::complex<float>* aterm_derivatives,
        double* hessian,
        double* gradient,
        double *residual);

    void Proxy_calibrate_finish(
         Proxy* p);

    void Proxy_calibrate_init_hessian_vector_product(Proxy* p);

    void Proxy_calibrate_hessian_vector_product1(
        Proxy* p,
        const unsigned int station_nr,
        const unsigned int subgrid_size,
        const unsigned int nr_stations,
        const unsigned int nr_time_slots,
        const unsigned int nr_terms,
        std::complex<float>* aterms,
        std::complex<float>* aterm_derivatives,
        float* parameter_vector);

    void Proxy_calibrate_update_hessian_vector_product2(
        Proxy* p,
        const unsigned int station_nr,
        const unsigned int subgrid_size,
        const unsigned int nr_stations,
        const unsigned int nr_time_slots,
        const unsigned int nr_terms,
        std::complex<float>* aterms,
        std::complex<float>* aterm_derivatives,
        float* parameter_vector);

    void Proxy_transform(
        Proxy* p,
        int direction,
        std::complex<float>* grid,
        unsigned int grid_nr_correlations,
        unsigned int grid_height,
        unsigned int grid_width);

    void Proxy_destroy(Proxy* p);

    void* Proxy_get_grid(
        Proxy* p,
        unsigned int nr_correlations,
        unsigned int grid_size);
}
