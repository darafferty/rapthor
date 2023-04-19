// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Proxy.h"
#include <functional>

/* Wrapper for member functions to catch exceptions, print a message and exit.
 * Used here in de C-interface, because C++ exceptions can not be propagated
 * further Uncaught exceptions will likely cause the program to crash
 *
 * Usage:
 *     Foo foo;
 *     ExitOnException(&Foo::bar, &foo, arg1, arg2);
 *
 * This will call foo.bar(arg1, arg2).
 *
 * To select a member function with overloads, use a cast to the required
 * function type, for example:
 *
 *     ExitOnException(static_cast<return_type(Foo::*)(arg1_type, arg2_type),
 * &foo, arg1, arg2);
 */

template <typename Fn, typename... Args>
typename std::result_of<Fn(Args...)>::type ExitOnException(Fn fn,
                                                           Args&&... args) {
  try {
    return std::mem_fn(fn)(std::forward<Args>(args)...);
  } catch (const std::exception& e) {
    std::cout << "IDG C-interface can not propagate exception" << std::endl;
    std::cout << e.what() << std::endl;
    std::cout << "Exiting..." << std::endl;
    exit(1);
  }
}

extern "C" {

void Proxy_gridding(struct Proxy* p, int kernel_size, int subgrid_size,
                    int nr_channels, int nr_baselines, int nr_timesteps,
                    int nr_correlations, int nr_timeslots, int nr_stations,
                    float* frequencies, std::complex<float>* visibilities,
                    idg::UVW<float>* uvw,
                    std::pair<unsigned int, unsigned int>* baselines,
                    std::complex<float>* aterms, unsigned int* aterm_offsets,
                    float* taper) {
  const std::array<size_t, 1> frequencies_shape{
      static_cast<size_t>(nr_channels)};
  const std::array<size_t, 2> uvw_shape{static_cast<size_t>(nr_baselines),
                                        static_cast<size_t>(nr_timesteps)};
  const std::array<size_t, 4> visibilities_shape{
      static_cast<size_t>(nr_baselines), static_cast<size_t>(nr_timesteps),
      static_cast<size_t>(nr_channels), static_cast<size_t>(nr_correlations)};
  const std::array<size_t, 1> baselines_shape{
      static_cast<size_t>(nr_baselines)};
  const std::array<size_t, 4> aterms_shape{
      static_cast<size_t>(nr_timeslots), static_cast<size_t>(nr_stations),
      static_cast<size_t>(subgrid_size), static_cast<size_t>(subgrid_size)};
  const std::array<size_t, 1> aterm_offsets_shape{
      static_cast<size_t>(nr_timeslots + 1)};
  const std::array<size_t, 2> taper_shape{static_cast<size_t>(subgrid_size),
                                          static_cast<size_t>(subgrid_size)};

  auto frequencies_span =
      aocommon::xt::CreateSpan(frequencies, frequencies_shape);
  auto uvw_span = aocommon::xt::CreateSpan(uvw, uvw_shape);
  auto visibilities_span =
      aocommon::xt::CreateSpan(visibilities, visibilities_shape);
  auto baselines_span = aocommon::xt::CreateSpan(baselines, baselines_shape);
  auto aterms_span = aocommon::xt::CreateSpan(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms),
      aterms_shape);
  auto aterm_offsets_span =
      aocommon::xt::CreateSpan(aterm_offsets, aterm_offsets_shape);
  auto taper_span = aocommon::xt::CreateSpan(taper, taper_shape);

  idg::Plan::Options options;
  options.mode = nr_correlations == 4 ? idg::Plan::Mode::FULL_POLARIZATION
                                      : idg::Plan::Mode::STOKES_I_ONLY;

  std::unique_ptr<idg::Plan> plan = ExitOnException(
      &idg::proxy::Proxy::make_plan, reinterpret_cast<idg::proxy::Proxy*>(p),
      kernel_size, frequencies_span, uvw_span, baselines_span,
      aterm_offsets_span, options);

  reinterpret_cast<idg::proxy::Proxy*>(p)->gridding(
      *plan, frequencies_span, visibilities_span, uvw_span, baselines_span,
      aterms_span, aterm_offsets_span, taper_span);
}

void Proxy_degridding(struct Proxy* p, int kernel_size, int subgrid_size,
                      int nr_channels, int nr_baselines, int nr_timesteps,
                      int nr_correlations, int nr_timeslots, int nr_stations,
                      float* frequencies, std::complex<float>* visibilities,
                      idg::UVW<float>* uvw,
                      std::pair<unsigned int, unsigned int>* baselines,
                      std::complex<float>* aterms, unsigned int* aterm_offsets,
                      float* taper) {
  const std::array<size_t, 1> frequencies_shape{
      static_cast<size_t>(nr_channels)};
  const std::array<size_t, 2> uvw_shape{static_cast<size_t>(nr_baselines),
                                        static_cast<size_t>(nr_timesteps)};
  const std::array<size_t, 4> visibilities_shape{
      static_cast<size_t>(nr_baselines), static_cast<size_t>(nr_timesteps),
      static_cast<size_t>(nr_channels), static_cast<size_t>(nr_correlations)};
  const std::array<size_t, 1> baselines_shape{
      static_cast<size_t>(nr_baselines)};
  const std::array<size_t, 4> aterms_shape{
      static_cast<size_t>(nr_timeslots), static_cast<size_t>(nr_stations),
      static_cast<size_t>(subgrid_size), static_cast<size_t>(subgrid_size)};
  const std::array<size_t, 1> aterm_offsets_shape{
      static_cast<size_t>(nr_timeslots + 1)};
  const std::array<size_t, 2> taper_shape{static_cast<size_t>(subgrid_size),
                                          static_cast<size_t>(subgrid_size)};

  auto frequencies_span =
      aocommon::xt::CreateSpan(frequencies, frequencies_shape);
  auto uvw_span = aocommon::xt::CreateSpan(uvw, uvw_shape);
  auto visibilities_span =
      aocommon::xt::CreateSpan(visibilities, visibilities_shape);
  auto baselines_span = aocommon::xt::CreateSpan(baselines, baselines_shape);
  auto aterms_span = aocommon::xt::CreateSpan(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms),
      aterms_shape);
  auto aterm_offsets_span =
      aocommon::xt::CreateSpan(aterm_offsets, aterm_offsets_shape);
  auto taper_span = aocommon::xt::CreateSpan(taper, taper_shape);

  idg::Plan::Options options;
  options.mode = nr_correlations == 4 ? idg::Plan::Mode::FULL_POLARIZATION
                                      : idg::Plan::Mode::STOKES_I_ONLY;

  std::unique_ptr<idg::Plan> plan = ExitOnException(
      &idg::proxy::Proxy::make_plan, reinterpret_cast<idg::proxy::Proxy*>(p),
      kernel_size, frequencies_span, uvw_span, baselines_span,
      aterm_offsets_span, options);

  reinterpret_cast<idg::proxy::Proxy*>(p)->degridding(
      *plan, frequencies_span, visibilities_span, uvw_span, baselines_span,
      aterms_span, aterm_offsets_span, taper_span);
}

void Proxy_init_cache(struct Proxy* p, unsigned int subgrid_size,
                      const float cell_size, float w_step, float* shift) {
  const std::array<float, 2> shift_array{shift[0], shift[1]};
  ExitOnException(&idg::proxy::Proxy::init_cache,
                  reinterpret_cast<idg::proxy::Proxy*>(p), subgrid_size,
                  cell_size, w_step, shift_array);
}

void Proxy_calibrate_init(struct Proxy* p, unsigned int kernel_size,
                          unsigned int subgrid_size,
                          unsigned int nr_channel_blocks,
                          unsigned int nr_channels_per_block,
                          unsigned int nr_baselines, unsigned int nr_timesteps,
                          unsigned int nr_timeslots, float* frequencies,
                          std::complex<float>* visibilities, float* weights,
                          float* uvw, unsigned int* baselines,
                          unsigned int* aterm_offsets, float* spheroidal) {
  const unsigned int nr_correlations = 4;
  const unsigned int nr_channels = nr_channel_blocks * nr_channels_per_block;
  idg::Array2D<float> frequencies_(frequencies, nr_channel_blocks,
                                   nr_channels_per_block);
  idg::Array4D<std::complex<float>> visibilities_(
      visibilities, nr_baselines, nr_timesteps, nr_channels, nr_correlations);
  idg::Array4D<float> weights_(reinterpret_cast<float*>(weights), nr_baselines,
                               nr_timesteps, nr_channels, nr_correlations);
  idg::Array2D<idg::UVW<float>> uvw_((idg::UVW<float>*)uvw, nr_baselines,
                                     nr_timesteps);
  const std::array<size_t, 1> baselines_shape{
      static_cast<size_t>(nr_baselines)};
  auto baselines_ = aocommon::xt::CreateSpan(
      reinterpret_cast<std::pair<unsigned int, unsigned int>*>(baselines),
      baselines_shape);
  idg::Array1D<unsigned int> aterm_offsets_(aterm_offsets, nr_timeslots + 1);
  idg::Array2D<float> spheroidal_(spheroidal, subgrid_size, subgrid_size);

  ExitOnException(&idg::proxy::Proxy::calibrate_init,
                  reinterpret_cast<idg::proxy::Proxy*>(p), kernel_size,
                  frequencies_, visibilities_, weights_, uvw_, baselines_,
                  aterm_offsets_, spheroidal_);
}

void Proxy_calibrate_update(
    struct Proxy* p, const unsigned int antenna_nr,
    const unsigned int nr_channel_blocks, const unsigned int subgrid_size,
    const unsigned int nr_antennas, const unsigned int nr_timeslots,
    const unsigned int nr_terms, std::complex<float>* aterms,
    std::complex<float>* aterm_derivatives, double* hessian, double* gradient,
    double* residual) {
  idg::Array5D<idg::Matrix2x2<std::complex<float>>> aterms_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterms),
      nr_channel_blocks, nr_timeslots, nr_antennas, subgrid_size, subgrid_size);
  idg::Array5D<idg::Matrix2x2<std::complex<float>>> aterm_derivatives_(
      reinterpret_cast<idg::Matrix2x2<std::complex<float>>*>(aterm_derivatives),
      nr_channel_blocks, nr_timeslots, nr_terms, subgrid_size, subgrid_size);
  idg::Array4D<double> hessian_(hessian, nr_channel_blocks, nr_timeslots,
                                nr_terms, nr_terms);
  idg::Array3D<double> gradient_(gradient, nr_channel_blocks, nr_timeslots,
                                 nr_terms);
  idg::Array1D<double> residual_(residual, nr_channel_blocks);
  reinterpret_cast<idg::proxy::Proxy*>(p)->calibrate_update(
      antenna_nr, aterms_, aterm_derivatives_, hessian_, gradient_, residual_);
}

void Proxy_calibrate_finish(struct Proxy* p) {
  ExitOnException(&idg::proxy::Proxy::calibrate_finish,
                  reinterpret_cast<idg::proxy::Proxy*>(p));
}

void Proxy_transform(struct Proxy* p, int direction) {
  ExitOnException(
      static_cast<void (idg::proxy::Proxy::*)(idg::DomainAtoDomainB)>(
          &idg::proxy::Proxy::transform),
      reinterpret_cast<idg::proxy::Proxy*>(p),
      direction ? idg::ImageDomainToFourierDomain
                : idg::FourierDomainToImageDomain);
}

void Proxy_destroy(struct Proxy* p) {
  delete reinterpret_cast<idg::proxy::Proxy*>(p);
}

void* Proxy_allocate_grid(struct Proxy* p, unsigned int nr_correlations,
                          unsigned int grid_size) {
  const unsigned int nr_w_layers = 1;
  auto grid = reinterpret_cast<idg::proxy::Proxy*>(p)->allocate_grid(
      nr_w_layers, nr_correlations, grid_size, grid_size);
  ExitOnException(&idg::proxy::Proxy::set_grid,
                  reinterpret_cast<idg::proxy::Proxy*>(p), grid);
  return grid->data();
}

void Proxy_set_grid(struct Proxy* p, std::complex<float>* grid_ptr,
                    unsigned int nr_w_layers, unsigned int nr_correlations,
                    unsigned int height, unsigned int width) {
  std::shared_ptr<idg::Grid> grid = std::shared_ptr<idg::Grid>(
      new idg::Grid(grid_ptr, nr_w_layers, nr_correlations, height, width));
  ExitOnException(&idg::proxy::Proxy::set_grid,
                  reinterpret_cast<idg::proxy::Proxy*>(p), grid);
}

void Proxy_get_final_grid(struct Proxy* p, std::complex<float>* grid_ptr,
                          unsigned int nr_w_layers,
                          unsigned int nr_correlations, unsigned int height,
                          unsigned int width) {
  std::shared_ptr<idg::Grid> grid =
      ExitOnException(&idg::proxy::Proxy::get_final_grid,
                      reinterpret_cast<idg::proxy::Proxy*>(p));
  if (grid_ptr) {
    assert(grid->get_w_dim() == nr_w_layers);
    assert(grid->get_z_dim() == nr_correlations);
    assert(grid->get_y_dim() == height);
    assert(grid->get_x_dim() == width);
    memcpy(grid_ptr, grid->data(), grid->bytes());
  }
}

}  // end extern "C"
