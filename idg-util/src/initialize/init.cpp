// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "init.h"

namespace idg {

// Function to compute spheroidal. Based on reference code by BvdT.
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

// TODO: make generic, not spheroidal specific
// TODO: use real-to-complex and complex-to-real FFT
void resize_spheroidal(float *__restrict__ spheroidal_in, unsigned int size_in,
                       float *__restrict__ spheroidal_out,
                       unsigned int size_out) {
  auto in_ft = new std::complex<float>[size_in * size_in];
  auto out_ft = new std::complex<float>[size_out * size_out];

  for (unsigned i = 0; i < size_in; i++) {
    for (unsigned j = 0; j < size_in; j++) {
      in_ft[i * size_in + j] = spheroidal_in[i * size_in + j];
    }
  }
  idg::fft2f(size_in, in_ft);

  int offset = int((size_out - size_in) / 2);

  for (unsigned i = 0; i < size_in; i++) {
    for (unsigned j = 0; j < size_in; j++) {
      out_ft[(i + offset) * size_out + (j + offset)] = in_ft[i * size_in + j];
    }
  }
  idg::ifft2f(size_out, out_ft);

  float s = 1.0f / (size_in * size_in);
  for (unsigned i = 0; i < size_out; i++) {
    for (unsigned j = 0; j < size_out; j++) {
      spheroidal_out[i * size_out + j] = out_ft[i * size_out + j].real() * s;
    }
  }

  delete[] in_ft;
  delete[] out_ft;
}

void init_example_aterms(Array4D<Matrix2x2<std::complex<float>>> &aterms) {
  unsigned int nr_timeslots = aterms.get_w_dim();
  unsigned int nr_stations = aterms.get_z_dim();
  unsigned int height = aterms.get_y_dim();
  unsigned int width = aterms.get_x_dim();

  for (unsigned t = 0; t < nr_timeslots; t++) {
    for (unsigned ant = 0; ant < nr_stations; ant++) {
      for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
          float scale = ((float)(t + 1) / nr_timeslots);
          std::complex<float> valueXX = std::complex<float>(scale * 1.0, 1.1);
          std::complex<float> valueXY = std::complex<float>(scale * 0.8, 0.9);
          std::complex<float> valueYX = std::complex<float>(scale * 0.6, 1.7);
          std::complex<float> valueYY = std::complex<float>(scale * 0.4, 0.5);
          const Matrix2x2<std::complex<float>> aterm = {valueXX, valueXY,
                                                        valueYX, valueYY};
          aterms(t, ant, y, x) = aterm;
        }
      }
    }
  }
}

/*
 * Memory-allocation is handled by Proxy
 */
Array1D<float> get_example_frequencies(proxy::Proxy &proxy,
                                       unsigned int nr_channels,
                                       float start_frequency,
                                       float frequency_increment) {
  using T = float;
  Array1D<T> frequencies = proxy.allocate_array1d<T>(nr_channels);

  for (unsigned chan = 0; chan < nr_channels; chan++) {
    frequencies(chan) = start_frequency + frequency_increment * chan;
  }

  return frequencies;
}

Array4D<std::complex<float>> get_dummy_visibilities(
    proxy::Proxy &proxy, unsigned int nr_baselines, unsigned int nr_timesteps,
    unsigned int nr_channels, unsigned int nr_correlations) {
  assert(nr_correlations == 2 || nr_correlations == 4);

  Array4D<std::complex<float>> visibilities =
      proxy.allocate_array4d<std::complex<float>>(nr_baselines, nr_timesteps,
                                                  nr_channels, nr_correlations);

// Set all visibilities
#pragma omp parallel for
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned time = 0; time < nr_timesteps; time++) {
      for (unsigned chan = 0; chan < nr_channels; chan++) {
        if (nr_correlations == 2) {
          visibilities(bl, time, chan, 0) = 1.0f;
          visibilities(bl, time, chan, 1) = 1.0f;
        } else if (nr_correlations == 4) {
          visibilities(bl, time, chan, 0) = 1.0f;
          visibilities(bl, time, chan, 1) = 0.0f;
          visibilities(bl, time, chan, 2) = 0.0f;
          visibilities(bl, time, chan, 3) = 1.0f;
        }
      }
    }
  }

  return visibilities;
}

Array4D<std::complex<float>> get_example_visibilities(
    proxy::Proxy &proxy, Array2D<UVW<float>> &uvw, Array1D<float> &frequencies,
    float image_size, unsigned int nr_correlations, unsigned int grid_size,
    unsigned int nr_point_sources, int max_pixel_offset,
    unsigned int random_seed, float amplitude) {
  unsigned int nr_baselines = uvw.get_y_dim();
  unsigned int nr_timesteps = uvw.get_x_dim();
  unsigned int nr_channels = frequencies.get_x_dim();

  Array4D<std::complex<float>> visibilities =
      proxy.allocate_array4d<std::complex<float>>(nr_baselines, nr_timesteps,
                                                  nr_channels, nr_correlations);

  if (max_pixel_offset == -1) {
    max_pixel_offset = grid_size / 2;
  }

  srand(random_seed);

  for (unsigned i = 0; i < nr_point_sources; i++) {
    float x_offset = static_cast<float>(random()) / RAND_MAX;
    float y_offset = static_cast<float>(random()) / RAND_MAX;
    x_offset = (x_offset * (max_pixel_offset)) - (max_pixel_offset / 2);
    y_offset = (y_offset * (max_pixel_offset)) - (max_pixel_offset / 2);

    add_pt_src(visibilities, uvw, frequencies, image_size, grid_size, x_offset,
               y_offset, amplitude);
  }

  return visibilities;
}

Array1D<std::pair<unsigned int, unsigned int>> get_example_baselines(
    proxy::Proxy &proxy, unsigned int nr_stations, unsigned int nr_baselines) {
  using T = std::pair<unsigned int, unsigned int>;
  Array1D<T> baselines = proxy.allocate_array1d<T>(nr_baselines);

  unsigned bl = 0;

  for (unsigned station1 = 0; station1 < nr_stations; station1++) {
    for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
      if (bl >= nr_baselines) {
        break;
      }
      baselines(bl) = std::pair<unsigned int, unsigned int>(station1, station2);
      bl++;
    }
  }

  return baselines;
}

Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
    proxy::Proxy &proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width) {
  using T = Matrix2x2<std::complex<float>>;
  Array4D<T> aterms =
      proxy.allocate_array4d<T>(nr_timeslots, nr_stations, height, width);

  for (unsigned t = 0; t < nr_timeslots; t++) {
    for (unsigned ant = 0; ant < nr_stations; ant++) {
      for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
          std::complex<float> valueXX = std::complex<float>(1.0, 0.0);
          std::complex<float> valueXY = std::complex<float>(0.0, 0.0);
          std::complex<float> valueYX = std::complex<float>(0.0, 0.0);
          std::complex<float> valueYY = std::complex<float>(1.0, 0.0);
          const Matrix2x2<std::complex<float>> aterm = {valueXX, valueXY,
                                                        valueYX, valueYY};
          aterms(t, ant, y, x) = aterm;
        }
      }
    }
  }

  return aterms;
}

Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
    proxy::Proxy &proxy, unsigned int nr_timeslots, unsigned int nr_stations,
    unsigned int height, unsigned int width) {
  using T = Matrix2x2<std::complex<float>>;
  Array4D<T> aterms =
      proxy.allocate_array4d<T>(nr_timeslots, nr_stations, height, width);

  for (unsigned t = 0; t < nr_timeslots; t++) {
    for (unsigned ant = 0; ant < nr_stations; ant++) {
      for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
          float scale = ((float)(t + 1) / nr_timeslots);
          std::complex<float> valueXX = std::complex<float>(scale * 1.0, 1.1);
          std::complex<float> valueXY = std::complex<float>(scale * 0.8, 0.9);
          std::complex<float> valueYX = std::complex<float>(scale * 0.6, 1.7);
          std::complex<float> valueYY = std::complex<float>(scale * 0.4, 0.5);
          const Matrix2x2<std::complex<float>> aterm = {valueXX, valueXY,
                                                        valueYX, valueYY};
          aterms(t, ant, y, x) = aterm;
        }
      }
    }
  }

  return aterms;
}

Array1D<unsigned int> get_example_aterms_offsets(proxy::Proxy &proxy,
                                                 unsigned int nr_timeslots,
                                                 unsigned int nr_timesteps) {
  using T = unsigned int;
  Array1D<T> aterms_offsets = proxy.allocate_array1d<T>(nr_timeslots + 1);

  for (unsigned time = 0; time < nr_timeslots; time++) {
    aterms_offsets(time) = time * (nr_timesteps / nr_timeslots);
  }

  aterms_offsets(nr_timeslots) = nr_timesteps;

  return aterms_offsets;
}

Array2D<float> get_example_spheroidal(proxy::Proxy &proxy, unsigned int height,
                                      unsigned int width) {
  using T = float;
  Array2D<T> spheroidal = proxy.allocate_array2d<T>(height, width);

  // Evaluate rows
  float y[height];
  for (unsigned i = 0; i < height; i++) {
    float tmp = fabs(-1 + i * 2.0f / float(height));
    y[i] = evaluate_spheroidal(tmp);
  }

  // Evaluate columns
  float x[width];
  for (unsigned i = 0; i < width; i++) {
    float tmp = fabs(-1 + i * 2.0f / float(width));
    x[i] = evaluate_spheroidal(tmp);
  }

  // Set values
  for (unsigned i = 0; i < height; i++) {
    for (unsigned j = 0; j < width; j++) {
      spheroidal(i, j) = y[i] * x[j];
    }
  }

  return spheroidal;
}

/*
 * Default memory allocation
 */
Array1D<float> get_example_frequencies(unsigned int nr_channels,
                                       float start_frequency,
                                       float frequency_increment) {
  Array1D<float> frequencies(nr_channels);

  for (unsigned chan = 0; chan < nr_channels; chan++) {
    frequencies(chan) = start_frequency + frequency_increment * chan;
  }

  return frequencies;
}

Array4D<std::complex<float>> get_dummy_visibilities(
    unsigned int nr_baselines, unsigned int nr_timesteps,
    unsigned int nr_channels, unsigned int nr_correlations) {
  assert(nr_correlations == 2 || nr_correlations == 4);

  Array4D<std::complex<float>> visibilities(nr_baselines, nr_timesteps,
                                            nr_channels, nr_correlations);

// Set all visibilities
#pragma omp parallel for
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned time = 0; time < nr_timesteps; time++) {
      for (unsigned chan = 0; chan < nr_channels; chan++) {
        if (nr_correlations == 2) {
          visibilities(bl, time, chan, 0) = 1.0f;
          visibilities(bl, time, chan, 1) = 1.0f;
        } else if (nr_correlations == 4) {
          visibilities(bl, time, chan, 0) = 1.0f;
          visibilities(bl, time, chan, 1) = 0.0f;
          visibilities(bl, time, chan, 2) = 0.0f;
          visibilities(bl, time, chan, 3) = 1.0f;
        }
      }
    }
  }

  return visibilities;
}

Array4D<std::complex<float>> get_example_visibilities(
    Array2D<UVW<float>> &uvw, Array1D<float> &frequencies, float image_size,
    unsigned int grid_size, unsigned int nr_correlations,
    unsigned int nr_point_sources, unsigned int max_pixel_offset,
    unsigned int random_seed, float amplitude) {
  unsigned int nr_baselines = uvw.get_y_dim();
  unsigned int nr_timesteps = uvw.get_x_dim();
  unsigned int nr_channels = frequencies.get_x_dim();

  Array4D<std::complex<float>> visibilities(nr_baselines, nr_timesteps,
                                            nr_channels, nr_correlations);
  std::fill_n(visibilities.data(), visibilities.size(),
              std::complex<float>{0.0, 0.0});

  srand(random_seed);

  for (unsigned i = 0; i < nr_point_sources; i++) {
    float x_offset = (random() * (max_pixel_offset)) - (max_pixel_offset / 2);
    float y_offset = (random() * (max_pixel_offset)) - (max_pixel_offset / 2);

    add_pt_src(visibilities, uvw, frequencies, image_size, grid_size, x_offset,
               y_offset, amplitude);
  }

  return visibilities;
}

Array1D<std::pair<unsigned int, unsigned int>> get_example_baselines(
    unsigned int nr_stations, unsigned int nr_baselines) {
  Array1D<std::pair<unsigned int, unsigned int>> baselines(nr_baselines);

  unsigned bl = 0;

  for (unsigned station1 = 0; station1 < nr_stations; station1++) {
    for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
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
                      const std::string &layout_file) {
  // Get data instance
  Data data(layout_file);

  // Determine the max baseline length for given grid_size
  float max_uv = data.compute_max_uv(grid_size, nr_channels);

  // Select only baselines up to max_uv meters long
  data.limit_max_baseline_length(max_uv);

  // Restrict the number of baselines to max_nr_baselines
  data.limit_nr_baselines(max_nr_baselines);

  // Return the resulting data
  return data;
}

Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
    unsigned int nr_timeslots, unsigned int nr_stations, unsigned int height,
    unsigned int width) {
  Array4D<Matrix2x2<std::complex<float>>> aterms(nr_timeslots, nr_stations,
                                                 height, width);
  const Matrix2x2<std::complex<float>> aterm = {1.0f, 0.0f, 0.0f, 1.0f};

  for (unsigned t = 0; t < nr_timeslots; t++) {
    for (unsigned ant = 0; ant < nr_stations; ant++) {
      for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
          aterms(t, ant, y, x) = aterm;
        }
      }
    }
  }

  return aterms;
}

Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
    unsigned int nr_timeslots, unsigned int nr_stations, unsigned int height,
    unsigned int width) {
  assert(height == width);
  Array4D<Matrix2x2<std::complex<float>>> aterms(nr_timeslots, nr_stations,
                                                 height, width);
  init_example_aterms(aterms);
  return aterms;
}

Array1D<unsigned int> get_example_aterms_offsets(unsigned int nr_timeslots,
                                                 unsigned int nr_timesteps) {
  Array1D<unsigned int> aterms_offsets(nr_timeslots + 1);

  for (unsigned time = 0; time < nr_timeslots; time++) {
    aterms_offsets(time) = time * (nr_timesteps / nr_timeslots);
  }

  aterms_offsets(nr_timeslots) = nr_timesteps;

  return aterms_offsets;
}

Array2D<float> get_identity_spheroidal(unsigned int height,
                                       unsigned int width) {
  Array2D<float> spheroidal(height, width);

  float value = 1.0;

  for (unsigned y = 0; y < height; y++) {
    for (unsigned x = 0; x < width; x++) {
      spheroidal(y, x) = value;
    }
  }

  return spheroidal;
}

Array2D<float> get_example_spheroidal(unsigned int height, unsigned int width) {
  Array2D<float> spheroidal(height, width);

  // Evaluate rows
  float y[height];
  for (unsigned i = 0; i < height; i++) {
    float tmp = fabs(-1 + i * 2.0f / float(height));
    y[i] = evaluate_spheroidal(tmp);
  }

  // Evaluate columns
  float x[width];
  for (unsigned i = 0; i < width; i++) {
    float tmp = fabs(-1 + i * 2.0f / float(width));
    x[i] = evaluate_spheroidal(tmp);
  }

  // Set values
  for (unsigned i = 0; i < height; i++) {
    for (unsigned j = 0; j < width; j++) {
      spheroidal(i, j) = y[i] * x[j];
    }
  }

  return spheroidal;
}

Array1D<float> get_zero_shift() {
  Array1D<float> shift(2);
  shift.zero();
  return shift;
}

void add_pt_src(Array4D<std::complex<float>> &visibilities,
                Array2D<UVW<float>> &uvw, Array1D<float> &frequencies,
                float image_size, unsigned int grid_size, float offset_x,
                float offset_y, float amplitude) {
  auto nr_baselines = visibilities.get_w_dim();
  auto nr_timesteps = visibilities.get_z_dim();
  auto nr_channels = visibilities.get_y_dim();
  auto nr_correlations = visibilities.get_x_dim();

  float l = offset_x * image_size / grid_size;
  float m = offset_y * image_size / grid_size;

  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    for (unsigned t = 0; t < nr_timesteps; t++) {
      for (unsigned c = 0; c < nr_channels; c++) {
        const double speed_of_light = 299792458.0;
        float u = (frequencies(c) / speed_of_light) * uvw(bl, t).u;
        float v = (frequencies(c) / speed_of_light) * uvw(bl, t).v;
        std::complex<float> value =
            amplitude *
            exp(std::complex<float>(0, -2 * M_PI * (u * l + v * m)));
        for (unsigned cor = 0; cor < nr_correlations; cor++) {
          visibilities(bl, t, c, cor) += value;
        }
      }
    }
  }
}

}  // namespace idg

#include "initc.h"
