// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include "Generic.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace cuda {

// Constructor
Generic::Generic(ProxyInfo info) : CUDA(info) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif
}

// Destructor
Generic::~Generic() {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif
}

/* High level routines */
void Generic::do_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array4D<std::complex<float>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  run_imaging(plan, frequencies, visibilities, uvw, baselines, *m_grid, aterms,
              aterms_offsets, spheroidal, ImagingMode::mode_gridding);
}

void Generic::do_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array4D<std::complex<float>>& visibilities, const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  run_imaging(plan, frequencies, visibilities, uvw, baselines, *m_grid, aterms,
              aterms_offsets, spheroidal, ImagingMode::mode_degridding);
}

void Generic::set_grid(std::shared_ptr<Grid> grid) {
  CUDA::set_grid(grid);
  if (m_disable_wtiling) {
    InstanceCUDA& device = get_device(0);
    cu::Stream& htodstream = device.get_htod_stream();
    cu::Context& context = get_device(0).get_context();
    d_grid_.reset(new cu::DeviceMemory(context, grid->bytes()));
    device.copy_htod(htodstream, *d_grid_, grid->data(), grid->bytes());
    htodstream.synchronize();
  }
}

std::shared_ptr<Grid> Generic::get_final_grid() {
  if (!m_disable_wtiling) {
    flush_wtiles();
  } else {
    InstanceCUDA& device = get_device(0);
    cu::Stream& dtohstream = device.get_dtoh_stream();
    device.copy_dtoh(dtohstream, m_grid->data(), *d_grid_, m_grid->bytes());
    dtohstream.synchronize();
  }
  return m_grid;
}

void Generic::check_grid() {
  if (m_disable_wtiling && !d_grid_) {
    throw std::runtime_error("device grid is not set, call set_grid first.");
  }
}

std::unique_ptr<Plan> Generic::make_plan(
    const int kernel_size, const Array1D<float>& frequencies,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets, Plan::Options options) {
  if (do_supports_wtiling() && !m_disable_wtiling) {
    options.w_step = m_cache_state.w_step;
    options.nr_w_layers = INT_MAX;
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, m_grid->get_y_dim(),
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterms_offsets, m_wtiles, options));
  } else {
    return Proxy::make_plan(kernel_size, frequencies, uvw, baselines,
                            aterms_offsets, options);
  }
}

void Generic::init_cache(int subgrid_size, float cell_size, float w_step,
                         const Array1D<float>& shift) {
  // Initialize cache
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);

  if (!m_disable_wtiling) {
    // Allocate wtiles on GPU
    InstanceCUDA& device = get_device(0);
    const cu::Context& context = get_device(0).get_context();

    // Compute the size of one tile
    const int nr_polarizations = m_grid->get_z_dim();
    int tile_size = m_tile_size + subgrid_size;
    size_t sizeof_tile =
        nr_polarizations * tile_size * tile_size * sizeof(std::complex<float>);

    // Initialize patches
    m_buffers_wtiling.d_patches.resize(m_nr_patches_batch);
    for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
      size_t sizeof_patch = nr_polarizations * m_patch_size * m_patch_size *
                            sizeof(std::complex<float>);
      m_buffers_wtiling.d_patches[i].reset(
          new cu::DeviceMemory(context, sizeof_patch));
    }

    // First, get the amount of free device memory
    size_t free_memory = device.get_free_memory();
    // Second, keep a safety margin
    free_memory *= 0.80;
    // Reserve 20% of the available GPU memory for each one of:
    // - The tiles: d_tiles (tile_size + subgrid_size)
    // - A subset of padded tiles: d_padded_tiles (tile_size + subgrid_size +
    // w_size)
    // - An FFT plan for the padded tiles
    m_nr_tiles = (free_memory * 0.2) / sizeof_tile;
    // The remainder should be enough for miscellaneous buffers:
    // - tile ids, tile coordinates, etc.

    // Allocate the tile buffers
    size_t sizeof_tiles = m_nr_tiles * sizeof_tile;
    m_buffers_wtiling.d_tiles.reset(
        new cu::DeviceMemory(context, sizeof_tiles));
    m_buffers_wtiling.d_padded_tiles.reset(
        new cu::DeviceMemory(context, sizeof_tiles));
    m_buffers_wtiling.h_tiles.reset(new cu::HostMemory(context, sizeof_tiles));

    // Initialize wtiles metadata
    m_wtiles = WTiles(m_nr_tiles, m_tile_size);
  }
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
