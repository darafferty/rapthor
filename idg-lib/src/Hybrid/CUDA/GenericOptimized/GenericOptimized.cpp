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

  // This proxy supports two modes:
  //  1) CPU-only W-Tiling
  //   This mode offloads the W-Tiling to the CPU proxy:
  //    - GPU: compute subgrids
  //    -  IO: copy subgrids to/from host
  //    - CPU: add/extract sugrids to/from grid using tiles
  //   It can be enabled by setting the environment flag DISABLE_WTILING_GPU.
  //  2) GPU-accelerated W-Tiling
  //   This mode performs most of the W-Tiling operations on the GPU:
  //    - GPU: compute subgrids, compute tiles, compute patches
  //    -  IO: copy patches to/from host
  //    - CPU: add/extract patches to/from grid

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
 * Gridding
 */
void GenericOptimized::do_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array4D<std::complex<float>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  // Since run_imaging supports both gridding and degridding, it expects
  // visibilities as non-const since degridding updates the visibilities.
  // We therefore need to const_cast it although it is used read-only during
  // gridding.
  auto& visibilities_ptr =
      const_cast<Array4D<std::complex<float>>&>(visibilities);
  run_imaging(plan, frequencies, visibilities_ptr, uvw, baselines, *m_grid,
              aterms, aterms_offsets, spheroidal, ImagingMode::mode_gridding);
}  // end do_gridding

void GenericOptimized::do_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array4D<std::complex<float>>& visibilities, const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  run_imaging(plan, frequencies, visibilities, uvw, baselines, *m_grid, aterms,
              aterms_offsets, spheroidal, ImagingMode::mode_degridding);
}  // end do_degridding

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
    init_buffers_wtiling(subgrid_size);
    m_wtiles = WTiles(m_nr_tiles, m_tile_size);
  } else {
    cpuProxy->init_cache(subgrid_size, cell_size, w_step, shift);
  }
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
