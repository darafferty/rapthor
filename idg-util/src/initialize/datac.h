extern "C" {

    idg::Data* DATA_init(
       unsigned int nr_stations_limit,
       unsigned int baseline_length_limit,
       const char *layout_file)
    {
        return new idg::Data(nr_stations_limit, baseline_length_limit, layout_file);
    }

    float DATA_compute_image_size(
        idg::Data* data,
        unsigned grid_size)
    {
        return data->compute_image_size(grid_size);
    }

    float DATA_compute_grid_size(
        idg::Data* data,
        float image_size)
    {
        return data->compute_grid_size(image_size);
    }

    float DATA_compute_max_uv(
        unsigned grid_size,
        float image_size)
    {
        return idg::Data::compute_max_uv(grid_size, image_size);
    }

    void DATA_filter_baselines(
        idg::Data* data,
        unsigned grid_size,
        float image_size)
    {
        data->filter_baselines(grid_size, image_size);
    }

    void DATA_limit_nr_baselines(
        idg::Data* data,
        unsigned int n)
    {
        data->limit_nr_baselines(n);
    }

    float DATA_get_nr_stations(
        idg::Data* data)
    {
        return data->get_nr_stations();
    }

    float DATA_get_nr_baselines(
        idg::Data* data)
    {
        return data->get_nr_baselines();
    }

    void DATA_get_frequencies(
        idg::Data* data,
        void* ptr,
        unsigned int nr_channels,
        float image_size,
        unsigned int channel_offset)
    {
        idg::Array1D<float> frequencies((float *) ptr, nr_channels);
        data->get_frequencies(frequencies, image_size, channel_offset);
    }

    void DATA_get_uvw(
        idg::Data* data,
        void* ptr,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int baseline_offset,
        unsigned int time_offset,
        float integration_time)
    {
        idg::Array2D<idg::UVW<float>> uvw((idg::UVW<float> *) ptr, nr_baselines, nr_timesteps);
        data->get_uvw(uvw, baseline_offset, time_offset, integration_time);
    }

    void DATA_shuffle_stations(
        idg::Data* data)
    {
        data->shuffle_stations();
    }

    void DATA_print_info(
        idg::Data* data)
    {
        data->print_info();
    }

}  // end extern "C"
