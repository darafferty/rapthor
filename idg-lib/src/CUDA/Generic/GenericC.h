// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

extern "C" {
typedef idg::proxy::cuda::Generic ProxyType;

ProxyType* CUDA_Generic_init() { return new ProxyType(); }

void CUDA_Generic_gridding(
    ProxyType* p, float w_step, float* shift, const float cell_size,
    unsigned int kernel_size, unsigned int subgrid_size, float* frequencies,
    unsigned int nr_channels, std::complex<float>* visibilities,
    unsigned int visibilities_nr_baselines,
    unsigned int visibilities_nr_timesteps,
    unsigned int visibilities_nr_channels,
    unsigned int visibilities_nr_correlations, float* uvw,
    unsigned int uvw_nr_baselines, unsigned int uvw_nr_timesteps,
    unsigned int uvw_nr_coordinates, unsigned int* baselines,
    unsigned int baselines_nr_baselines, unsigned int baselines_two,
    std::complex<float>* grid, unsigned int grid_nr_correlations,
    unsigned int grid_height, unsigned int grid_width,
    std::complex<float>* aterms, unsigned int aterms_nr_timeslots,
    unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
    unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
    unsigned int* aterms_offsets,
    unsigned int aterms_offsets_nr_timeslots_plus_one, float* spheroidal,
    unsigned int spheroidal_height, unsigned int spheroidal_width) {
  p->gridding(w_step, shift, cell_size, kernel_size, subgrid_size, frequencies,
              nr_channels, visibilities, visibilities_nr_baselines,
              visibilities_nr_timesteps, visibilities_nr_channels,
              visibilities_nr_correlations, uvw, uvw_nr_baselines,
              uvw_nr_timesteps, uvw_nr_coordinates, baselines,
              baselines_nr_baselines, baselines_two, grid, grid_nr_correlations,
              grid_height, grid_width, aterms, aterms_nr_timeslots,
              aterms_nr_stations, aterms_aterm_height, aterms_aterm_width,
              aterms_nr_correlations, aterms_offsets,
              aterms_offsets_nr_timeslots_plus_one, spheroidal,
              spheroidal_height, spheroidal_width);
}

void CUDA_Generic_degridding(
    ProxyType* p, float w_step, float* shift, const float cell_size,
    unsigned int kernel_size, unsigned int subgrid_size, float* frequencies,
    unsigned int nr_channels, std::complex<float>* visibilities,
    unsigned int visibilities_nr_baselines,
    unsigned int visibilities_nr_timesteps,
    unsigned int visibilities_nr_channels,
    unsigned int visibilities_nr_correlations, float* uvw,
    unsigned int uvw_nr_baselines, unsigned int uvw_nr_timesteps,
    unsigned int uvw_nr_coordinates, unsigned int* baselines,
    unsigned int baselines_nr_baselines, unsigned int baselines_two,
    std::complex<float>* grid, unsigned int grid_nr_correlations,
    unsigned int grid_height, unsigned int grid_width,
    std::complex<float>* aterms, unsigned int aterms_nr_timeslots,
    unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
    unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
    unsigned int* aterms_offsets,
    unsigned int aterms_offsets_nr_timeslots_plus_one, float* spheroidal,
    unsigned int spheroidal_height, unsigned int spheroidal_width) {
  p->degridding(w_step, shift, cell_size, kernel_size, subgrid_size,
                frequencies, nr_channels, visibilities,
                visibilities_nr_baselines, visibilities_nr_timesteps,
                visibilities_nr_channels, visibilities_nr_correlations, uvw,
                uvw_nr_baselines, uvw_nr_timesteps, uvw_nr_coordinates,
                baselines, baselines_nr_baselines, baselines_two, grid,
                grid_nr_correlations, grid_height, grid_width, aterms,
                aterms_nr_timeslots, aterms_nr_stations, aterms_aterm_height,
                aterms_aterm_width, aterms_nr_correlations, aterms_offsets,
                aterms_offsets_nr_timeslots_plus_one, spheroidal,
                spheroidal_height, spheroidal_width);
}

void CUDA_Generic_transform(ProxyType* p, int direction,
                            std::complex<float>* grid,
                            unsigned int grid_nr_correlations,
                            unsigned int grid_height, unsigned int grid_width) {
  if (direction != 0) {
    p->transform(idg::ImageDomainToFourierDomain, grid, grid_nr_correlations,
                 grid_height, grid_width);
  } else {
    p->transform(idg::FourierDomainToImageDomain, grid, grid_nr_correlations,
                 grid_height, grid_width);
  }
}

void CUDA_Generic_destroy(ProxyType* p) { delete p; }
}  // end extern "C"
