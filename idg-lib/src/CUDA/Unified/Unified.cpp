// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include "Unified.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace cuda {

// Constructor
Unified::Unified(ProxyInfo info) : Generic(info) {
#if defined(DEBUG)
  cout << "Unified::" << __func__ << endl;
#endif

  // Increase the fraction of reserved memory
  set_fraction_reserved(0.4);

  // Enable unified memory
  enable_unified_memory();
}

// Destructor
Unified::~Unified() {
#if defined(DEBUG)
  std::cout << "Unified::" << __func__ << std::endl;
#endif
}

void Unified::do_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array4D<std::complex<float>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Unified::" << __func__ << std::endl;
#endif
  if (!m_use_unified_memory) {
    throw std::runtime_error("Unified memory needs to be enabled!");
  }

  auto grid_ptr = m_grid.get();
  if (m_enable_tiling) {
    auto nr_polarizations = m_grid->get_z_dim();
    auto height = m_grid->get_y_dim();
    auto width = m_grid->get_x_dim();
    grid_ptr =
        new idg::Grid(m_grid_tiled->data(), 1, nr_polarizations, height, width);
  }
  Generic::run_imaging(plan, frequencies, visibilities, uvw, baselines,
                       *grid_ptr, aterms, aterms_offsets, spheroidal,
                       ImagingMode::mode_gridding);
}  // end gridding

void Unified::do_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array4D<std::complex<float>>& visibilities, const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Unified::" << __func__ << std::endl;
#endif
  if (!m_use_unified_memory) {
    throw std::runtime_error("Unified memory needs to be enabled!");
  }

  auto grid_ptr = m_grid.get();
  if (m_enable_tiling) {
    auto nr_polarizations = m_grid->get_z_dim();
    auto height = m_grid->get_y_dim();
    auto width = m_grid->get_x_dim();
    grid_ptr =
        new idg::Grid(m_grid_tiled->data(), 1, nr_polarizations, height, width);
  }
  Generic::run_imaging(plan, frequencies, visibilities, uvw, baselines,
                       *grid_ptr, aterms, aterms_offsets, spheroidal,
                       ImagingMode::mode_degridding);
}  // end degridding

void Unified::do_transform(idg::DomainAtoDomainB direction) {
  InstanceCUDA& device = get_device(0);
  int nr_polarizations = m_grid->get_z_dim();
  int grid_size = m_grid->get_x_dim();
  int tile_size = device.get_tile_size_grid();
  if (m_enable_tiling) {
    device.tile_backward(nr_polarizations, grid_size, tile_size, *m_grid_tiled,
                         *m_grid);
  }
  Generic::set_grid(m_grid);
  Generic::do_transform(direction);
  m_grid = Generic::get_final_grid();
  if (m_enable_tiling) {
    device.tile_forward(nr_polarizations, grid_size, tile_size, *m_grid,
                        *m_grid_tiled);
  }
}

void Unified::set_grid(std::shared_ptr<Grid> grid) {
  m_grid = grid;

  if (m_enable_tiling) {
    InstanceCUDA& device = get_device(0);

    auto nr_w_layers = grid->get_w_dim();
    auto nr_polarizations = grid->get_z_dim();
    auto grid_height = grid->get_y_dim();
    auto grid_width = grid->get_x_dim();
    assert(grid_height == grid_width);
    auto grid_size = grid_width;
    auto tile_size = device.get_tile_size_grid();
    const cu::Context& context = device.get_context();
    cu::UnifiedMemory* u_grid_tiled =
        new cu::UnifiedMemory(context, m_grid->bytes());
    assert(nr_w_layers == 1);
    auto nr_tiles_1d = grid_size / tile_size;
    auto* grid_tiled = new Array5D<std::complex<float>>(
        *u_grid_tiled, nr_tiles_1d, nr_tiles_1d, nr_polarizations, tile_size,
        tile_size);
    m_grid_tiled.reset(grid_tiled);
    device.tile_forward(nr_polarizations, grid_size, tile_size, *grid,
                        *m_grid_tiled);
  }
}

std::shared_ptr<Grid> Unified::get_final_grid() {
  if (m_enable_tiling) {
    InstanceCUDA& device = get_device(0);
    auto nr_polarizations = m_grid->get_z_dim();
    auto grid_size = m_grid->get_x_dim();
    auto tile_size = device.get_tile_size_grid();
    device.tile_backward(nr_polarizations, grid_size, tile_size, *m_grid_tiled,
                         *m_grid);
  }

  return m_grid;
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
