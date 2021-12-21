// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_INIT_H_
#define IDG_INIT_H_

#include "Data.h"

#include "idg-util-config.h"
#include "idg-common.h"  // idg data types
#include "idg-fft.h"

namespace idg {
/*
 * Data generator
 */
Data get_example_data(unsigned int max_nr_baselines, unsigned int grid_size,
                      float integration_time, unsigned int nr_channels,
                      const std::string& layout_file);

/*
 * Memory-allocation is handled by Proxy
 */
Array1D<float> get_example_frequencies(
    proxy::Proxy& proxy, unsigned int nr_channels,
    float start_frequency = Data::start_frequency,
    float frequency_increment = Data::frequency_increment);

Array4D<std::complex<float>> get_dummy_visibilities(
    proxy::Proxy& proxy, unsigned int nr_baselines, unsigned int nr_timesteps,
    unsigned int nr_channels, unsigned int nr_correlations);

Array4D<std::complex<float>> get_example_visibilities(
    proxy::Proxy& proxy, Array2D<UVW<float>>& uvw, Array1D<float>& frequencies,
    float image_size, unsigned int nr_correlations, unsigned int grid_size,
    unsigned int nr_polarizations, unsigned int nr_point_sources = 4,
    unsigned int max_pixel_offset = -1, unsigned int random_seed = 2,
    float amplitude = 1);

Array1D<std::pair<unsigned int, unsigned int>> get_example_baselines(
    proxy::Proxy& proxy, unsigned int nr_stations, unsigned int nr_baselines);

Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width);

Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width);

Array1D<unsigned int> get_example_aterms_offsets(proxy::Proxy& proxy,
                                                 unsigned int nr_timeslots,
                                                 unsigned int nr_timesteps);

Array2D<float> get_example_spheroidal(proxy::Proxy& proxy, unsigned int height,
                                      unsigned int width);

Array1D<float> get_zero_shift();

/*
 * Default memory allocation
 */
Array1D<float> get_example_frequencies(
    unsigned int nr_channels, float start_frequency = Data::start_frequency,
    float frequency_increment = Data::frequency_increment);

Array4D<std::complex<float>> get_dummy_visibilities(
    unsigned int nr_baselines, unsigned int nr_timesteps,
    unsigned int nr_channels, unsigned int nr_correlations);

Array4D<std::complex<float>> get_example_visibilities(
    Array2D<UVW<float>>& uvw, Array1D<float>& frequencies, float image_size,
    unsigned int grid_size, unsigned int nr_correlations,
    unsigned int nr_point_sources = 4, unsigned int max_pixel_offset = -1,
    unsigned int random_seed = 2, float amplitude = 1);

Array1D<std::pair<unsigned int, unsigned int>> get_example_baselines(
    unsigned int nr_stations, unsigned int nr_baselines);

Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
    unsigned int nr_timeslots, unsigned int nr_stations, unsigned int height,
    unsigned int width);

Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
    unsigned int nr_timeslots, unsigned int nr_stations, unsigned int height,
    unsigned int width);

Array1D<unsigned int> get_example_aterms_offsets(unsigned int nr_timeslots,
                                                 unsigned int nr_timesteps);

Array2D<float> get_identity_spheroidal(unsigned int height, unsigned int width);

Array2D<float> get_example_spheroidal(unsigned int height, unsigned int width);

float evaluate_spheroidal(float nu);

void add_pt_src(Array4D<std::complex<float>>& visibilities,
                Array2D<UVW<float>>& uvw, Array1D<float>& frequencies,
                float image_size, unsigned int grid_size, float x, float y,
                float amplitude);

}  // namespace idg

#endif
