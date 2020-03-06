#ifndef IDG_INIT_H_
#define IDG_INIT_H_

#include "Data.h"

#include "idg-util-config.h"
#include "idg-common.h" // idg data types
#include "idg-fft.h"


namespace idg {

    /*
     * Memory-allocation is handled by Proxy
     */
    Array1D<float> get_example_frequencies(
        proxy::Proxy& proxy,
        unsigned int nr_channels,
        float start_frequency = Data::start_frequency,
        float frequency_increment = Data::frequency_increment);

    Array3D<Visibility<std::complex<float>>> get_dummy_visibilities(
        proxy::Proxy& proxy,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_channels);

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        proxy::Proxy& proxy,
        Array2D<UVW<float>> &uvw,
        Array1D<float> &frequencies,
        float        image_size,
        unsigned int grid_size,
        unsigned int nr_point_sources = 4,
        unsigned int max_pixel_offset = -1,
        unsigned int random_seed = 2,
        float        amplitude = 1);

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        proxy::Proxy& proxy,
        unsigned int nr_stations,
        unsigned int nr_baselines);

    Array2D<UVW<float>> get_example_uvw(
        proxy::Proxy& proxy,
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time = Data::integration_time);

    Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array1D<unsigned int> get_example_aterms_offsets(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_timesteps);

    Array2D<float> get_example_spheroidal(
        proxy::Proxy& proxy,
        unsigned int height,
        unsigned int width);

    Array1D<float> get_zero_shift();

    /*
     * Default memory allocation
     */
    Array1D<float> get_example_frequencies(
        unsigned int nr_channels,
        float start_frequency = Data::start_frequency,
        float frequency_increment = Data::frequency_increment);

    Array3D<Visibility<std::complex<float>>> get_dummy_visibilities(
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_channels);

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        Array2D<UVW<float>> &uvw,
        Array1D<float> &frequencies,
        float        image_size,
        unsigned int grid_size,
        unsigned int nr_point_sources = 4,
        unsigned int max_pixel_offset = -1,
        unsigned int random_seed = 2,
        float        amplitude = 1);

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        unsigned int nr_stations,
        unsigned int nr_baselines);

    Array2D<UVW<float>> get_example_uvw(
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time = Data::integration_time);

    Array3D<std::complex<float>> get_zero_grid(
        unsigned int nr_correlations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array1D<unsigned int> get_example_aterms_offsets(
        unsigned int nr_timeslots,
        unsigned int nr_timesteps);

    Array2D<float> get_identity_spheroidal(
        unsigned int height,
        unsigned int width);

    Array2D<float> get_example_spheroidal(
        unsigned int height,
        unsigned int width);

    float evaluate_spheroidal(float nu);

    void add_pt_src(
        Array3D<Visibility<std::complex<float>>> &visibilities,
        Array2D<UVW<float>> &uvw,
        Array1D<float> &frequencies,
        float          image_size,
        unsigned int   grid_size,
        float          x,
        float          y,
        float          amplitude);

} // namespace idg

#endif
