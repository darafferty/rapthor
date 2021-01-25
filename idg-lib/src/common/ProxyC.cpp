// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Proxy.h"

extern "C" {

void Proxy_gridding(
    struct Proxy* p, float w_step, float* shift, const float cell_size,
    unsigned int kernel_size, unsigned int subgrid_size, float* frequencies,
    unsigned int nr_channels, std::complex<float>* visibilities,
    unsigned int visibilities_nr_baselines,
    unsigned int visibilities_nr_timesteps,
    unsigned int visibilities_nr_channels,
    unsigned int visibilities_nr_correlations, float* uvw,
    unsigned int uvw_nr_baselines, unsigned int uvw_nr_timesteps,
    unsigned int uvw_nr_coordinates, unsigned int* baselines,
    unsigned int baselines_nr_baselines, unsigned int baselines_two,
    std::complex<float>* aterms, unsigned int aterms_nr_timeslots,
    unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
    unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
    unsigned int* aterms_offsets,
    unsigned int aterms_offsets_nr_timeslots_plus_one, float* spheroidal,
    unsigned int spheroidal_height, unsigned int spheroidal_width) {
  idg::Array1D<float> shift_(shift, 3);
  idg::Array1D<float> frequencies_(frequencies, nr_channels);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_(
      (idg::Visibility<std::complex<float>>*)visibilities, visibilities_nr_baselines,
      visibilities_nr_timesteps, visibilities_nr_channels);
  idg::Array2D<idg::UVW<float>> uvw_((idg::UVW<float>*)uvw, uvw_nr_baselines,
                           uvw_nr_timesteps);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines_(
      (std::pair<unsigned int, unsigned int>*)baselines,
      baselines_nr_baselines);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(
      (idg::Matrix2x2<std::complex<float>>*)aterms, aterms_nr_timeslots,
      aterms_nr_stations, aterms_aterm_height, aterms_aterm_width);
  idg::Array1D<unsigned int> aterms_offsets_(aterms_offsets,
                                        aterms_offsets_nr_timeslots_plus_one);
  idg::Array2D<float> spheroidal_(spheroidal, spheroidal_height, spheroidal_width);

  reinterpret_cast<idg::proxy::Proxy*>(p)->gridding(w_step, shift_, cell_size, kernel_size, subgrid_size, frequencies_,
           visibilities_, uvw_, baselines_, aterms_, aterms_offsets_,
           spheroidal_);
}

void Proxy_degridding(
    struct Proxy* p, float w_step, float* shift, const float cell_size,
    unsigned int kernel_size, unsigned int subgrid_size, float* frequencies,
    unsigned int nr_channels, std::complex<float>* visibilities,
    unsigned int visibilities_nr_baselines,
    unsigned int visibilities_nr_timesteps,
    unsigned int visibilities_nr_channels,
    unsigned int visibilities_nr_correlations, float* uvw,
    unsigned int uvw_nr_baselines, unsigned int uvw_nr_timesteps,
    unsigned int uvw_nr_coordinates, unsigned int* baselines,
    unsigned int baselines_nr_baselines, unsigned int baselines_two,
    std::complex<float>* aterms, unsigned int aterms_nr_timeslots,
    unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
    unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
    unsigned int* aterms_offsets,
    unsigned int aterms_offsets_nr_timeslots_plus_one, float* spheroidal,
    unsigned int spheroidal_height, unsigned int spheroidal_width) {
  idg::Array1D<float> shift_(shift, 3);
  idg::Array1D<float> frequencies_(frequencies, nr_channels);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_(
      (idg::Visibility<std::complex<float>>*)visibilities, visibilities_nr_baselines,
      visibilities_nr_timesteps, visibilities_nr_channels);
  idg::Array2D<idg::UVW<float>> uvw_((idg::UVW<float>*)uvw, uvw_nr_baselines,
                           uvw_nr_timesteps);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines_(
      (std::pair<unsigned int, unsigned int>*)baselines,
      baselines_nr_baselines);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(
      (idg::Matrix2x2<std::complex<float>>*)aterms, aterms_nr_timeslots,
      aterms_nr_stations, aterms_aterm_height, aterms_aterm_width);
  idg::Array1D<unsigned int> aterms_offsets_(aterms_offsets,
                                        aterms_offsets_nr_timeslots_plus_one);
  idg::Array2D<float> spheroidal_(spheroidal, spheroidal_height, spheroidal_width);

  reinterpret_cast<idg::proxy::Proxy*>(p)->degridding(w_step, shift_, cell_size, kernel_size, subgrid_size, frequencies_,
             visibilities_, uvw_, baselines_, aterms_, aterms_offsets_,
             spheroidal_);
}

void Proxy_calibrate_init(struct Proxy* p, float w_step, float* shift,
                          const float cell_size, unsigned int kernel_size,
                          unsigned int subgrid_size, unsigned int nr_channels,
                          unsigned int nr_baselines, unsigned int nr_timesteps,
                          unsigned int nr_timeslots,
                          unsigned int nr_correlations,
                          unsigned int grid_height, unsigned int grid_width,
                          float* frequencies, std::complex<float>* visibilities,
                          float* weights, float* uvw, unsigned int* baselines,
                          std::complex<float>* grid,
                          unsigned int* aterms_offsets, float* spheroidal) {
  idg::Array1D<float> shift_(shift, 3);
  idg::Array1D<float> frequencies_(frequencies, nr_channels);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_(
      (idg::Visibility<std::complex<float>>*)visibilities, nr_baselines,
      nr_timesteps, nr_channels);
  idg::Array3D<idg::Visibility<float>> weights_(
      (idg::Visibility<float>*)weights, nr_baselines, nr_timesteps,
      nr_channels);
  idg::Array2D<idg::UVW<float>> uvw_((idg::UVW<float>*)uvw, nr_baselines,
                                     nr_timesteps);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines_(
      (std::pair<unsigned int, unsigned int>*)baselines, nr_baselines);
  idg::Grid grid_(grid, 1, nr_correlations, grid_height, grid_width);
  idg::Array1D<unsigned int> aterms_offsets_(aterms_offsets, nr_timeslots + 1);
  idg::Array2D<float> spheroidal_(spheroidal, subgrid_size, subgrid_size);

  reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_init(
      kernel_size, frequencies_,
      visibilities_, weights_, uvw_, baselines_, aterms_offsets_,
      spheroidal_);
}

void Proxy_calibrate_update(
    struct Proxy* p, const unsigned int antenna_nr, const unsigned int subgrid_size,
    const unsigned int nr_antennas, const unsigned int nr_timeslots,
    const unsigned int nr_terms, std::complex<float>* aterms,
    std::complex<float>* aterm_derivatives, double* hessian, double* gradient,
    double* residual) {
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms),
      nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives),
      nr_timeslots, nr_terms, subgrid_size, subgrid_size);
  idg::Array3D<double> hessian_(hessian, nr_timeslots, nr_terms, nr_terms);
  idg::Array2D<double> gradient_(gradient, nr_timeslots, nr_terms);
  reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_update(
      antenna_nr, aterms_, aterm_derivatives_, hessian_, gradient_, *residual);
}

void Proxy_calibrate_finish(struct Proxy* p) {
  reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_finish();
}

void Proxy_calibrate_init_hessian_vector_product(struct Proxy* p) {
  reinterpret_cast<idg::proxy::Proxy*>(p)
      ->calibrate_init_hessian_vector_product();
}

void Proxy_calibrate_hessian_vector_product1(
    struct Proxy* p, const unsigned int antenna_nr, const unsigned int subgrid_size,
    const unsigned int nr_antennas, const unsigned int nr_timeslots,
    const unsigned int nr_terms, std::complex<float>* aterms,
    std::complex<float>* aterm_derivatives, float* parameter_vector) {
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms),
      nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives),
      nr_timeslots, nr_terms, subgrid_size, subgrid_size);
  idg::Array2D<float> parameter_vector_(parameter_vector, nr_timeslots,
                                        nr_terms);
  reinterpret_cast<idg::proxy::Proxy*>(p)
      ->calibrate_update_hessian_vector_product1(
          antenna_nr, aterms_, aterm_derivatives_, parameter_vector_);
}

void Proxy_calibrate_hessian_vector_product2(
    struct Proxy* p, const unsigned int antenna_nr, const unsigned int subgrid_size,
    const unsigned int nr_antennas, const unsigned int nr_timeslots,
    const unsigned int nr_terms, std::complex<float>* aterms,
    std::complex<float>* aterm_derivatives, float* parameter_vector) {
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms),
      nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives),
      nr_timeslots, nr_terms, subgrid_size, subgrid_size);
  idg::Array2D<float> parameter_vector_(parameter_vector, nr_timeslots,
                                        nr_terms);
  reinterpret_cast<idg::proxy::Proxy*>(p)
      ->calibrate_update_hessian_vector_product2(
          antenna_nr, aterms_, aterm_derivatives_, parameter_vector_);
}

void Proxy_transform(struct Proxy* p, int direction) {
  if (direction != 0) {
    reinterpret_cast<idg::proxy::Proxy*>(p)->transform(
        idg::ImageDomainToFourierDomain);
  } else {
    reinterpret_cast<idg::proxy::Proxy*>(p)->transform(
        idg::FourierDomainToImageDomain);
  }
}

void Proxy_destroy(struct Proxy* p) { delete reinterpret_cast<idg::proxy::Proxy*>(p); }

void* Proxy_allocate_grid(struct Proxy* p, unsigned int nr_correlations,
                          unsigned int grid_size) {
  const unsigned int nr_w_layers = 1;
  auto grid = reinterpret_cast<idg::proxy::Proxy*>(p)->allocate_grid(
      nr_w_layers, nr_correlations, grid_size, grid_size);
  reinterpret_cast<idg::proxy::Proxy*>(p)->set_grid(grid);
  return grid->data();
}

void Proxy_set_grid(struct Proxy* p, std::complex<float>* grid_ptr,
                    unsigned int nr_w_layers, unsigned int nr_correlations,
                    unsigned int height, unsigned int width) {
  std::shared_ptr<idg::Grid> grid = std::shared_ptr<idg::Grid>(
      new idg::Grid(grid_ptr, nr_w_layers, nr_correlations, height, width));
  reinterpret_cast<idg::proxy::Proxy*>(p)->set_grid(grid);
}

void Proxy_get_grid(struct Proxy* p, std::complex<float>* grid_ptr,
                    unsigned int nr_w_layers, unsigned int nr_correlations,
                    unsigned int height, unsigned int width) {
  std::shared_ptr<idg::Grid> grid =
      reinterpret_cast<idg::proxy::Proxy*>(p)->get_grid();
  assert(grid->get_w_dim() == nr_w_layers);
  assert(grid->get_z_dim() == nr_correlations);
  assert(grid->get_y_dim() == height);
  assert(grid->get_x_dim() == width);
  memcpy(grid_ptr, grid->data(), grid->bytes());
}

}  // end extern "C"
