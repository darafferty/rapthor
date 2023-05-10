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
aocommon::xt::Span<float, 1> get_example_frequencies(
    proxy::Proxy& proxy, unsigned int nr_channels,
    float start_frequency = Data::start_frequency,
    float frequency_increment = Data::frequency_increment);

aocommon::xt::Span<std::complex<float>, 4> get_dummy_visibilities(
    proxy::Proxy& proxy, unsigned int nr_baselines, unsigned int nr_timesteps,
    unsigned int nr_channels, unsigned int nr_correlations);

aocommon::xt::Span<std::complex<float>, 4> get_example_visibilities(
    proxy::Proxy& proxy, aocommon::xt::Span<UVW<float>, 2>& uvw,
    aocommon::xt::Span<float, 1>& frequencies, float image_size,
    unsigned int nr_correlations, unsigned int grid_size,
    unsigned int nr_point_sources = 4, unsigned int max_pixel_offset = 0,
    unsigned int random_seed = 2, float amplitude = 1);

aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>
get_example_baselines(proxy::Proxy& proxy, unsigned int nr_stations,
                      unsigned int nr_baselines);

aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4> get_identity_aterms(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width);

aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4> get_example_aterms(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width);

aocommon::xt::Span<unsigned int, 1> get_example_aterm_offsets(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_timesteps);

// Initialize taper with all ones
aocommon::xt::Span<float, 2> get_identity_taper(proxy::Proxy& proxy,
                                                unsigned int height,
                                                unsigned int width);

// Initialize taper with prolate spheroidal
aocommon::xt::Span<float, 2> get_example_taper(proxy::Proxy& proxy,
                                               unsigned int height,
                                               unsigned int width);

/*
 * Default memory allocation
 */
void init_example_frequencies(
    aocommon::xt::Span<float, 1>& frequencies,
    float start_frequency = Data::start_frequency,
    float frequency_increment = Data::frequency_increment);

void init_example_visibilities(
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    aocommon::xt::Span<UVW<float>, 2>& uvw,
    aocommon::xt::Span<float, 1>& frequencies, float image_size,
    unsigned int grid_size, unsigned int nr_point_sources = 4,
    unsigned int max_pixel_offset = 0, unsigned int random_seed = 2,
    float amplitude = 1);

void init_dummy_visibilities(
    aocommon::xt::Span<std::complex<float>, 4>& visibilities);

xt::xtensor<std::pair<unsigned int, unsigned int>, 1> get_example_baselines(
    unsigned int nr_stations, unsigned int nr_baselines);

xt::xtensor<Matrix2x2<std::complex<float>>, 4> get_identity_aterms(
    unsigned int nr_timeslots, unsigned int nr_stations, unsigned int height,
    unsigned int width);

xt::xtensor<Matrix2x2<std::complex<float>>, 4> get_example_aterms(
    unsigned int nr_timeslots, unsigned int nr_stations, unsigned int height,
    unsigned int width);

xt::xtensor<unsigned int, 1> get_example_aterm_offsets(
    unsigned int nr_timeslots, unsigned int nr_timesteps);

// Initialize taper with all ones
void init_identity_taper(aocommon::xt::Span<float, 2>& taper);

// Initialize taper with prolate spheroidal
void init_example_taper(aocommon::xt::Span<float, 2>& taper);

float evaluate_spheroidal(float nu);

void add_pt_src(aocommon::xt::Span<std::complex<float>, 4>& visibilities,
                const aocommon::xt::Span<UVW<float>, 2>& uvw,
                const aocommon::xt::Span<float, 1>& frequencies,
                float image_size, unsigned int grid_size, float offset_x,
                float offset_y, float amplitude);

void init_identity_aterms(
    aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms);

void init_example_aterms(
    aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms);

void init_example_aterm_offsets(
    aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    unsigned int nr_timesteps);

void init_example_baselines(
    aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>& baselines,
    unsigned int nr_stations);

}  // namespace idg

#endif
