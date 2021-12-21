// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

extern "C" {

void utils_init_identity_spheroidal(void *ptr, int subgrid_size) {
  idg::Array2D<float> spheroidal =
      idg::get_identity_spheroidal(subgrid_size, subgrid_size);
  memcpy(ptr, spheroidal.data(), spheroidal.bytes());
}

void utils_init_example_spheroidal(void *ptr, int subgrid_size) {
  idg::Array2D<float> spheroidal =
      idg::get_example_spheroidal(subgrid_size, subgrid_size);
  memcpy(ptr, spheroidal.data(), spheroidal.bytes());
}

void utils_init_example_frequencies(void *ptr, int nr_channels) {
  idg::Array1D<float> frequencies = idg::get_example_frequencies(nr_channels);
  memcpy(ptr, frequencies.data(), frequencies.bytes());
}

void utils_init_dummy_visibilities(void *ptr, int nr_baselines,
                                   int nr_timesteps, int nr_channels,
                                   int nr_correlations) {
  idg::Array4D<std::complex<float>> visibilities = idg::get_dummy_visibilities(
      nr_baselines, nr_timesteps, nr_channels, nr_correlations);
  memcpy(ptr, visibilities.data(), visibilities.bytes());
}

void utils_add_pt_src(float x, float y, float amplitude, int nr_baselines,
                      int nr_timesteps, int nr_channels, int nr_correlations,
                      float image_size, int grid_size, void *uvw,
                      void *frequencies, void *visibilities) {
  typedef idg::UVW<float> UVWType;
  idg::Array4D<std::complex<float>> visibilities_(
      reinterpret_cast<std::complex<float> *>(visibilities), nr_baselines,
      nr_timesteps, nr_channels, nr_correlations);
  idg::Array2D<UVWType> uvw_((UVWType *)uvw, nr_baselines, nr_timesteps);
  idg::Array1D<float> frequencies_((float *)frequencies, nr_channels);
  idg::add_pt_src(visibilities_, uvw_, frequencies_, image_size, grid_size, x,
                  y, amplitude);
}

void utils_init_identity_aterms(void *ptr, int nr_timeslots, int nr_stations,
                                int subgrid_size, int nr_correlations) {
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
      idg::get_identity_aterms(nr_timeslots, nr_stations, subgrid_size,
                               subgrid_size);
  memcpy(ptr, aterms.data(), aterms.bytes());
}

void utils_init_example_aterms(void *ptr, int nr_timeslots, int nr_stations,
                               int subgrid_size, int nr_correlations) {
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
      idg::get_example_aterms(nr_timeslots, nr_stations, subgrid_size,
                              subgrid_size);
  memcpy(ptr, aterms.data(), aterms.bytes());
}

void utils_init_example_aterms_offset(void *ptr, int nr_timeslots,
                                      int nr_timesteps) {
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(nr_timeslots, nr_timesteps);
  memcpy(ptr, aterms_offsets.data(), aterms_offsets.bytes());
}

void utils_init_example_baselines(void *ptr, int nr_stations,
                                  int nr_baselines) {
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      idg::get_example_baselines(nr_stations, nr_baselines);
  memcpy(ptr, baselines.data(), baselines.bytes());
}

}  // end extern "C"
