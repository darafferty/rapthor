// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

extern "C" {

void utils_init_identity_taper(void* ptr, int subgrid_size) {
  const std::array<size_t, 2> taper_shape{static_cast<size_t>(subgrid_size),
                                          static_cast<size_t>(subgrid_size)};
  auto taper =
      aocommon::xt::CreateSpan(reinterpret_cast<float*>(ptr), taper_shape);
  idg::init_identity_taper(taper);
}

void utils_init_example_taper(void* ptr, int subgrid_size) {
  const std::array<size_t, 2> taper_shape{static_cast<size_t>(subgrid_size),
                                          static_cast<size_t>(subgrid_size)};
  auto taper =
      aocommon::xt::CreateSpan(reinterpret_cast<float*>(ptr), taper_shape);
  idg::init_example_taper(taper);
}

void utils_init_example_frequencies(void* ptr, int nr_channels) {
  const std::array<size_t, 1> frequencies_shape{
      static_cast<size_t>(nr_channels)};
  auto frequencies = aocommon::xt::CreateSpan(reinterpret_cast<float*>(ptr),
                                              frequencies_shape);
  idg::init_example_frequencies(frequencies);
}

void utils_init_dummy_visibilities(void* ptr, int nr_baselines,
                                   int nr_timesteps, int nr_channels,
                                   int nr_correlations) {
  const std::array<size_t, 4> visibilities_shape{
      static_cast<size_t>(nr_baselines), static_cast<size_t>(nr_timesteps),
      static_cast<size_t>(nr_channels), static_cast<size_t>(nr_correlations)};
  auto visibilities = aocommon::xt::CreateSpan(
      reinterpret_cast<std::complex<float>*>(ptr), visibilities_shape);
  idg::init_dummy_visibilities(visibilities);
}

void utils_add_pt_src(float x, float y, float amplitude, int nr_baselines,
                      int nr_timesteps, int nr_channels, int nr_correlations,
                      float image_size, int grid_size, void* uvw,
                      void* frequencies, void* visibilities) {
  const std::array<size_t, 1> frequencies_shape{
      static_cast<size_t>(nr_channels)};
  auto frequencies_span = aocommon::xt::CreateSpan(
      reinterpret_cast<float*>(frequencies), frequencies_shape);
  const std::array<size_t, 2> uvw_shape{static_cast<size_t>(nr_baselines),
                                        static_cast<size_t>(nr_timesteps)};
  auto uvw_span = aocommon::xt::CreateSpan(
      reinterpret_cast<idg::UVW<float>*>(uvw), uvw_shape);
  const std::array<size_t, 4> visibilities_shape{
      static_cast<size_t>(nr_baselines), static_cast<size_t>(nr_timesteps),
      static_cast<size_t>(nr_channels), static_cast<size_t>(nr_correlations)};
  auto visibilities_span = aocommon::xt::CreateSpan(
      reinterpret_cast<std::complex<float>*>(visibilities), visibilities_shape);
  idg::add_pt_src(visibilities_span, uvw_span, frequencies_span, image_size,
                  grid_size, x, y, amplitude);
}

void utils_init_identity_aterms(void* ptr, int nr_timeslots, int nr_stations,
                                int subgrid_size, int nr_correlations) {
  const std::array<size_t, 4> aterms_shape{
      static_cast<size_t>(nr_timeslots), static_cast<size_t>(nr_stations),
      static_cast<size_t>(subgrid_size), static_cast<size_t>(subgrid_size)};
  using T = idg::Matrix2x2<std::complex<float>>;
  auto aterms =
      aocommon::xt::CreateSpan(reinterpret_cast<T*>(ptr), aterms_shape);
  idg::init_identity_aterms(aterms);
}

void utils_init_example_aterms(void* ptr, int nr_timeslots, int nr_stations,
                               int subgrid_size, int nr_correlations) {
  const std::array<size_t, 4> aterms_shape{
      static_cast<size_t>(nr_timeslots), static_cast<size_t>(nr_stations),
      static_cast<size_t>(subgrid_size), static_cast<size_t>(subgrid_size)};
  using Aterm = idg::Matrix2x2<std::complex<float>>;
  auto aterms =
      aocommon::xt::CreateSpan(reinterpret_cast<Aterm*>(ptr), aterms_shape);
  idg::init_example_aterms(aterms);
}

void utils_init_example_aterm_offsets(void* ptr, int nr_timeslots,
                                      int nr_timesteps) {
  const std::array<size_t, 1> aterm_offsets_shape{
      static_cast<size_t>(nr_timeslots) + 1};
  auto aterm_offsets = aocommon::xt::CreateSpan(
      reinterpret_cast<unsigned int*>(ptr), aterm_offsets_shape);
  idg::init_example_aterm_offsets(aterm_offsets, nr_timesteps);
}

void utils_init_example_baselines(void* ptr, int nr_stations,
                                  int nr_baselines) {
  const std::array<size_t, 1> baselines_shape{
      static_cast<size_t>(nr_baselines)};
  auto baselines = aocommon::xt::CreateSpan(
      reinterpret_cast<std::pair<unsigned int, unsigned int>*>(ptr),
      baselines_shape);
  idg::init_example_baselines(baselines, nr_stations);
}

}  // end extern "C"
