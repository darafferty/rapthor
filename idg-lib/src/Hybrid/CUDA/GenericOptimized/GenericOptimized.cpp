// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "GenericOptimized.h"

#include <cuda.h>

#include <algorithm>  // max_element
#include <mutex>
#include <csignal>

#include "InstanceCUDA.h"
#include "kernels/KernelGridder.cuh"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;

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
}

// Destructor
GenericOptimized::~GenericOptimized() {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  delete cpuProxy;
}

/*
 * Plan
 */
std::unique_ptr<Plan> GenericOptimized::make_plan(
    const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    Plan::Options options) {
  if (!m_disable_wtiling && !m_disable_wtiling_gpu) {
    options.w_step = m_cache_state.w_step;
    options.nr_w_layers = INT_MAX;
    // The number of channels per channel group should not exceed the number of
    // threads in a thread block for the CUDA gridder kernel
    options.max_nr_channels_per_subgrid =
        options.max_nr_channels_per_subgrid
            ? min(options.max_nr_channels_per_subgrid,
                  KernelGridder::block_size_x)
            : KernelGridder::block_size_x;
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, grid_size,
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterm_offsets, m_wtiles, options));
  } else {
    // Defer call to cpuProxy
    return cpuProxy->make_plan(kernel_size, frequencies, uvw, baselines,
                               aterm_offsets, options);
  }
}

/*
 * Gridding
 */
void GenericOptimized::do_gridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  // Since run_imaging supports both gridding and degridding, it expects
  // visibilities as non-const since degridding updates the visibilities.
  // We therefore need to const_cast it although it is used read-only during
  // gridding.
  auto& visibilities_ptr =
      const_cast<aocommon::xt::Span<std::complex<float>, 4>&>(visibilities);
  run_imaging(plan, frequencies, visibilities_ptr, uvw, baselines, get_grid(),
              aterms, aterm_offsets, taper, ImagingMode::mode_gridding);
}  // end do_gridding

void GenericOptimized::do_degridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  run_imaging(plan, frequencies, visibilities, uvw, baselines, get_grid(),
              aterms, aterm_offsets, taper, ImagingMode::mode_degridding);
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
void GenericOptimized::set_grid(
    aocommon::xt::Span<std::complex<float>, 4>& grid) {
  cpuProxy->set_grid(grid);
  CUDA::set_grid(grid);
}

aocommon::xt::Span<std::complex<float>, 4>& GenericOptimized::get_final_grid() {
  if (!m_disable_wtiling_gpu) {
    flush_wtiles();
    return get_grid();
  } else {
    // Defer call to cpuProxy
    return cpuProxy->get_final_grid();
  }
}

/*
 * Cache
 */
void GenericOptimized::init_cache(int subgrid_size, float cell_size,
                                  float w_step,
                                  const std::array<float, 2>& shift) {
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
