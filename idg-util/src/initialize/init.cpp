// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "init.h"

#include <xtensor/xview.hpp>
namespace {
// Function to compute spheroidal.
// Based on libreSpheroidal function in CASA
// https://github.com/radio-astro/casa/blob/4ebd5b1508a5d31b74e7b5f6b89313368d30b9ef/code/synthesis/TransformMachines/Utils.cc#L776
float evaluate_spheroidal(float nu) {
  float P[2][5] = {
      {8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
      {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
  float Q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                   {1.0000000e0, 9.599102e-1, 2.918724e-1}};

  int part;
  float end;
  if (nu >= 0.0 && nu < 0.75) {
    part = 0;
    end = 0.75f;
  } else if (nu >= 0.75 && nu <= 1.00) {
    part = 1;
    end = 1.0f;
  } else {
    return 0.0f;
  }

  float nusq = nu * nu;
  float delnusq = nusq - end * end;
  float delnusqPow = delnusq;
  float top = P[part][0];
  for (auto k = 1; k < 5; k++) {
    top += P[part][k] * delnusqPow;
    delnusqPow *= delnusq;
  }

  float bot = Q[part][0];
  delnusqPow = delnusq;
  for (auto k = 1; k < 3; k++) {
    bot += Q[part][k] * delnusqPow;
    delnusqPow *= delnusq;
  }

  if (bot == 0.0f) {
    return 0.0f;
  } else {
    return (1.0 - nusq) * (top / bot);
  }
}
}  // namespace
namespace idg {

/*
 * Memory-allocation is handled by Proxy
 */
aocommon::xt::Span<float, 1> get_example_frequencies(
    proxy::Proxy& proxy, unsigned int nr_channels, float start_frequency,
    float frequency_increment) {
  auto frequencies = proxy.allocate_span<float, 1>({nr_channels});
  init_example_frequencies(frequencies, start_frequency, frequency_increment);
  return frequencies;
}

aocommon::xt::Span<std::complex<float>, 4> get_dummy_visibilities(
    proxy::Proxy& proxy, unsigned int nr_baselines, unsigned int nr_timesteps,
    unsigned int nr_channels, unsigned int nr_correlations) {
  assert(nr_correlations == 2 || nr_correlations == 4);
  auto visibilities = proxy.allocate_span<std::complex<float>, 4>(
      {nr_baselines, nr_timesteps, nr_channels, nr_correlations});
  init_dummy_visibilities(visibilities);
  return visibilities;
}

aocommon::xt::Span<std::complex<float>, 4> get_example_visibilities(
    proxy::Proxy& proxy, aocommon::xt::Span<UVW<float>, 2>& uvw,
    aocommon::xt::Span<float, 1>& frequencies, float image_size,
    unsigned int nr_correlations, unsigned int grid_size,
    unsigned int nr_point_sources, unsigned int max_pixel_offset,
    unsigned int random_seed, float amplitude) {
  const size_t nr_baselines = uvw.shape(0);
  const size_t nr_timesteps = uvw.shape(1);
  const size_t nr_channels = frequencies.shape(0);
  auto visibilities = proxy.allocate_span<std::complex<float>, 4>(
      {nr_baselines, nr_timesteps, nr_channels, nr_correlations});
  init_example_visibilities(visibilities, uvw, frequencies, image_size,
                            grid_size, nr_point_sources, max_pixel_offset,
                            random_seed, amplitude);
  return visibilities;
}

aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>
get_example_baselines(proxy::Proxy& proxy, unsigned int nr_stations,
                      unsigned int nr_baselines) {
  using T = std::pair<unsigned int, unsigned int>;
  auto baselines = proxy.allocate_span<T, 1>({nr_baselines});
  init_example_baselines(baselines, nr_stations);
  return baselines;
}

aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4> get_identity_aterms(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width) {
  using T = Matrix2x2<std::complex<float>>;
  auto aterms =
      proxy.allocate_span<T, 4>({nr_timeslots, nr_stations, height, width});
  init_identity_aterms(aterms);
  return aterms;
}

aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4> get_example_aterms(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width) {
  using T = Matrix2x2<std::complex<float>>;
  auto aterms =
      proxy.allocate_span<T, 4>({nr_timeslots, nr_stations, height, width});
  init_example_aterms(aterms);
  return aterms;
}

aocommon::xt::Span<unsigned int, 1> get_example_aterm_offsets(
    proxy::Proxy& proxy, unsigned int nr_timeslots, unsigned int nr_timesteps) {
  using T = unsigned int;
  auto aterm_offsets = proxy.allocate_span<T, 1>({nr_timeslots + 1});
  init_example_aterm_offsets(aterm_offsets, nr_timesteps);
  return aterm_offsets;
}

aocommon::xt::Span<float, 2> get_identity_taper(proxy::Proxy& proxy,
                                                unsigned int height,
                                                unsigned int width) {
  using T = float;
  auto taper = proxy.allocate_span<T, 2>({height, width});
  init_identity_taper(taper);
  return taper;
}

aocommon::xt::Span<float, 2> get_example_taper(proxy::Proxy& proxy,
                                               unsigned int height,
                                               unsigned int width) {
  using T = float;
  auto taper = proxy.allocate_span<T, 2>({height, width});
  init_example_taper(taper);
  return taper;
}

/*
 * Default memory allocation
 */
void init_example_frequencies(aocommon::xt::Span<float, 1>& frequencies,
                              float start_frequency,
                              float frequency_increment) {
  const size_t nr_channels = frequencies.size();

  for (unsigned chan = 0; chan < nr_channels; chan++) {
    frequencies(chan) = start_frequency + frequency_increment * chan;
  }
}

void init_example_visibilities(
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    aocommon::xt::Span<UVW<float>, 2>& uvw,
    aocommon::xt::Span<float, 1>& frequencies, float image_size,
    unsigned int grid_size, unsigned int nr_point_sources,
    unsigned int max_pixel_offset, unsigned int random_seed, float amplitude) {
  const size_t nr_baselines = visibilities.shape(0);
  const size_t nr_timesteps = visibilities.shape(1);
  const size_t nr_channels = visibilities.shape(2);
  const size_t nr_correlations = visibilities.shape(3);
  assert(nr_correlations == 2 || nr_correlations == 4);
  assert(uvw.shape(0) == nr_baselines);
  assert(uvw.shape(1) == nr_timesteps);
  assert(frequencies.size() == nr_channels);

  if (max_pixel_offset == 0) {
    max_pixel_offset = grid_size / 2;
  }

  srand(random_seed);

  for (size_t i = 0; i < nr_point_sources; i++) {
    float x_offset = static_cast<float>(random()) / RAND_MAX;
    float y_offset = static_cast<float>(random()) / RAND_MAX;
    x_offset = (x_offset * (max_pixel_offset)) - (max_pixel_offset / 2);
    y_offset = (y_offset * (max_pixel_offset)) - (max_pixel_offset / 2);

    add_pt_src(visibilities, uvw, frequencies, image_size, grid_size, x_offset,
               y_offset, amplitude);
  }
}

void init_dummy_visibilities(
    aocommon::xt::Span<std::complex<float>, 4>& visibilities) {
  const size_t nr_baselines = visibilities.shape(0);
  const size_t nr_correlations = visibilities.shape(3);
  assert(nr_correlations == 2 || nr_correlations == 4);

// Set all visibilities
#pragma omp parallel for
  for (size_t bl = 0; bl < nr_baselines; bl++) {
    if (nr_correlations == 2) {
      xt::view(visibilities, bl, xt::all(), xt::all(), xt::all()) = 1.0f;
    } else if (nr_correlations == 4) {
      xt::view(visibilities, bl, xt::all(), xt::all(), 0) = 1.0f;
      xt::view(visibilities, bl, xt::all(), xt::all(), 1) = 0.0f;
      xt::view(visibilities, bl, xt::all(), xt::all(), 2) = 0.0f;
      xt::view(visibilities, bl, xt::all(), xt::all(), 3) = 1.0f;
    }
  }
}

xt::xtensor<std::pair<unsigned int, unsigned int>, 1> get_example_baselines(
    unsigned int nr_stations, unsigned int nr_baselines) {
  const std::array<size_t, 1> shape{nr_baselines};
  xt::xtensor<std::pair<unsigned int, unsigned int>, 1> baselines(shape);

  size_t bl = 0;

  for (size_t station1 = 0; station1 < nr_stations; station1++) {
    for (size_t station2 = station1 + 1; station2 < nr_stations; station2++) {
      if (bl >= nr_baselines) {
        break;
      }
      baselines(bl) = std::pair<unsigned int, unsigned int>(station1, station2);
      bl++;
    }
  }

  return baselines;
}

Data get_example_data(unsigned int max_nr_baselines, unsigned int grid_size,
                      float integration_time, unsigned int nr_channels,
                      const std::string& layout_file) {
  // Get data instance
  Data data(layout_file);

  // Determine the max baseline length for given grid_size
  const float max_uv = data.compute_max_uv(grid_size, nr_channels);

  // Select only baselines up to max_uv meters long
  data.limit_max_baseline_length(max_uv);

  // Restrict the number of baselines to max_nr_baselines
  data.limit_nr_baselines(max_nr_baselines);

  // Return the resulting data
  return data;
}

void add_pt_src(aocommon::xt::Span<std::complex<float>, 4>& visibilities,
                const aocommon::xt::Span<UVW<float>, 2>& uvw,
                const aocommon::xt::Span<float, 1>& frequencies,
                float image_size, unsigned int grid_size, float offset_x,
                float offset_y, float amplitude) {
  const size_t nr_baselines = visibilities.shape(0);
  const size_t nr_timesteps = visibilities.shape(1);
  const size_t nr_channels = visibilities.shape(2);

  const float l = offset_x * image_size / grid_size;
  const float m = offset_y * image_size / grid_size;

  for (size_t bl = 0; bl < nr_baselines; bl++) {
    for (size_t t = 0; t < nr_timesteps; t++) {
      for (size_t c = 0; c < nr_channels; c++) {
        const double speed_of_light = 299792458.0;
        const float u = (frequencies(c) / speed_of_light) * uvw(bl, t).u;
        const float v = (frequencies(c) / speed_of_light) * uvw(bl, t).v;
        const std::complex<float> value =
            amplitude *
            std::exp(std::complex<float>(0, -2 * M_PI * (u * l + v * m)));
        xt::view(visibilities, bl, t, c, xt::all()) += value;
      }
    }
  }
}

void init_identity_aterms(
    aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms) {
  const std::complex<float> valueXX{1.0f, 0.0f};
  const std::complex<float> valueXY{0.0f, 0.0f};
  const std::complex<float> valueYX{0.0f, 0.0f};
  const std::complex<float> valueYY{1.0f, 0.0f};
  const Matrix2x2<std::complex<float>> aterm{valueXX, valueXY, valueYX,
                                             valueYY};
  aterms.fill(aterm);
}

void init_example_aterms(
    aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms) {
  const size_t nr_timeslots = aterms.shape(0);

  for (size_t t = 0; t < nr_timeslots; t++) {
    const float scale = ((float)(t + 1) / nr_timeslots);
    std::complex<float> valueXX{scale * 1.0f, 1.1f};
    std::complex<float> valueXY{scale * 0.8f, 0.9f};
    std::complex<float> valueYX{scale * 0.6f, 1.7f};
    std::complex<float> valueYY{scale * 0.4f, 0.5f};
    const Matrix2x2<std::complex<float>> aterm{valueXX, valueXY, valueYX,
                                               valueYY};
    xt::view(aterms, t, xt::all(), xt::all(), xt::all()) = aterm;
  }
}

void init_example_aterm_offsets(
    aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    unsigned int nr_timesteps) {
  const size_t nr_timeslots = aterm_offsets.shape(0) - 1;

  for (size_t time = 0; time < nr_timeslots; time++) {
    aterm_offsets(time) = time * (nr_timesteps / nr_timeslots);
  }

  aterm_offsets(nr_timeslots) = nr_timesteps;
}

void init_identity_taper(aocommon::xt::Span<float, 2>& taper) {
  taper.fill(1.0f);
}

void init_example_taper(aocommon::xt::Span<float, 2>& taper) {
  const size_t height = taper.shape(0);
  const size_t width = taper.shape(1);

  // Evaluate rows
  float y[height];
  for (unsigned i = 0; i < height; i++) {
    float tmp = fabs(-1 + i * 2.0f / float(height));
    y[i] = ::evaluate_spheroidal(tmp);
  }

  // Evaluate columns
  float x[width];
  for (unsigned i = 0; i < width; i++) {
    float tmp = fabs(-1 + i * 2.0f / float(width));
    x[i] = ::evaluate_spheroidal(tmp);
  }

  // Set values
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      taper(i, j) = y[i] * x[j];
    }
  }
}

void init_example_baselines(
    aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>& baselines,
    unsigned int nr_stations) {
  const size_t nr_baselines = baselines.size();
  size_t bl = 0;

  for (size_t station1 = 0; station1 < nr_stations; station1++) {
    for (size_t station2 = station1 + 1; station2 < nr_stations; station2++) {
      if (bl >= nr_baselines) {
        break;
      }
      baselines(bl) = std::pair<unsigned int, unsigned int>(station1, station2);
      bl++;
    }
  }
}

}  // namespace idg

#include "initc.h"
