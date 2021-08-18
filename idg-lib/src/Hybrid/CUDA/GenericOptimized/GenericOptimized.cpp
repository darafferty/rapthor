// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "GenericOptimized.h"

#include <cuda.h>
#include <cudaProfiler.h>

#include <algorithm>  // max_element
#include <mutex>
#include <csignal>

#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace hybrid {

// Constructor
GenericOptimized::GenericOptimized() : CUDA(default_info()) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  // Initialize cpu proxy
  cpuProxy = new idg::proxy::cpu::Optimized();

  // Set W-Tiling GPU
  set_disable_wtiling_gpu(getenv("DISABLE_WTILING_GPU"));

  omp_set_nested(true);

  cuProfilerStart();
}

// Destructor
GenericOptimized::~GenericOptimized() {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  delete cpuProxy;
  cuProfilerStop();
}

/*
 * Plan
 */
std::unique_ptr<Plan> GenericOptimized::make_plan(
    const int kernel_size, const Array1D<float>& frequencies,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets, Plan::Options options) {
  if (!m_disable_wtiling && !m_disable_wtiling_gpu) {
    options.w_step = m_cache_state.w_step;
    options.nr_w_layers = INT_MAX;
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, m_grid->get_y_dim(),
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterms_offsets, m_wtiles, options));
  } else {
    // Defer call to cpuProxy
    return cpuProxy->make_plan(kernel_size, frequencies, uvw, baselines,
                               aterms_offsets, options);
  }
}

/*
 * FFT
 */
void GenericOptimized::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
  std::cout << "Transform direction: " << direction << std::endl;
#endif

  if (!m_disable_wtiling_gpu) {
    cpuProxy->set_grid(get_final_grid());
  }
  cpuProxy->transform(direction);
}  // end transform

/*
 * Grid
 */
void GenericOptimized::set_grid(std::shared_ptr<Grid> grid) {
  cpuProxy->set_grid(grid);
  CUDA::set_grid(grid);
}

std::shared_ptr<Grid> GenericOptimized::get_final_grid() {
  if (!m_disable_wtiling_gpu) {
    flush_wtiles();
    return m_grid;
  } else {
    // Defer call to cpuProxy
    return cpuProxy->get_final_grid();
  }
}

/*
 * Cache
 */
void GenericOptimized::init_cache(int subgrid_size, float cell_size,
                                  float w_step, const Array1D<float>& shift) {
  // Initialize cache
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);

  if (!m_disable_wtiling && !m_disable_wtiling_gpu) {
    // Allocate wtiles on GPU
    InstanceCUDA& device = get_device(0);
    const cu::Context& context = get_device(0).get_context();

    // Compute the size of one tile
    int tile_size = m_tile_size + subgrid_size;
    size_t sizeof_tile =
        NR_CORRELATIONS * tile_size * tile_size * sizeof(idg::float2);

    // Initialize patches
    m_buffers_wtiling.d_patches.resize(m_nr_patches_batch);
    for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
      size_t sizeof_patch = NR_CORRELATIONS * m_patch_size * m_patch_size *
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
  } else {
    cpuProxy->init_cache(subgrid_size, cell_size, w_step, shift);
  }
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
