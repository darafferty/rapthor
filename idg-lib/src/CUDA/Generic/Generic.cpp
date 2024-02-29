// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include <cudawrappers/cu.hpp>

#include "Generic.h"
#include "InstanceCUDA.h"
#include "kernels/KernelGridder.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {

// Constructor
Generic::Generic() : CUDA() {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  // This proxy supports two modes:
  //  1) Legacy
  //    GPU-only gridding/degridding without W-Tiling
  //  2) W-Tiling:
  //    Gridding/degridding and W-Tiling on the GPU using the
  //    same W-Tiling as in GenericOptimized.

  // Mode 1 (legacy)
  // m_disable_wtiling = true;

  // Mode 2 (W-Tiling)
  m_disable_wtiling = false;
}

// Destructor
Generic::~Generic() {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif
}

/* High level routines */
void Generic::do_gridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  run_imaging(plan, frequencies, visibilities, uvw, baselines, get_grid(),
              aterms, aterm_offsets, taper, ImagingMode::mode_gridding);
}

void Generic::do_degridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  run_imaging(plan, frequencies, visibilities, uvw, baselines, get_grid(),
              aterms, aterm_offsets, taper, ImagingMode::mode_degridding);
}

void Generic::set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) {
  const size_t nr_w_layers = grid.shape(0);
  assert(nr_w_layers == 1);
  const size_t grid_size = grid.shape(2);
  assert(grid.shape(3) == grid_size);
  const size_t sizeof_grid = grid.size() * sizeof(*grid.data());

  CUDA::set_grid(grid);
  if (m_disable_wtiling) {
    InstanceCUDA& device = get_device();
    cu::Stream& htodstream = device.get_htod_stream();
    d_grid_.reset(new cu::DeviceMemory(sizeof_grid, CU_MEMORYTYPE_DEVICE));
    htodstream.memcpyHtoDAsync(*d_grid_, grid.data(), sizeof_grid);
  }
}

aocommon::xt::Span<std::complex<float>, 4>& Generic::get_final_grid() {
  if (!m_disable_wtiling) {
    flush_wtiles();
  }

  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);

  if (m_disable_wtiling) {
    InstanceCUDA& device = get_device();
    cu::Stream& dtohstream = device.get_dtoh_stream();
    const size_t sizeof_grid = get_grid().size() * sizeof(*get_grid().data());
    dtohstream.memcpyDtoHAsync(get_grid().data(), *d_grid_, sizeof_grid);
  }
  return get_grid();
}

std::unique_ptr<Plan> Generic::make_plan(
    const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    Plan::Options options) {
  if (do_supports_wtiling() && !m_disable_wtiling) {
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    options.w_step = m_cache_state.w_step;
    options.nr_w_layers = std::numeric_limits<int>::max();
    options.max_nr_channels_per_subgrid =
        options.max_nr_channels_per_subgrid
            ? min(options.max_nr_channels_per_subgrid,
                  KernelGridder::kBlockSizeX)
            : KernelGridder::kBlockSizeX;
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, grid_size,
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterm_offsets, m_wtiles, options));
  } else {
    return Proxy::make_plan(kernel_size, frequencies, uvw, baselines,
                            aterm_offsets, options);
  }
}

void Generic::init_cache(int subgrid_size, float cell_size, float w_step,
                         const std::array<float, 2>& shift) {
  // Initialize cache
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);

  if (!m_disable_wtiling) {
    init_buffers_wtiling(subgrid_size);
    m_wtiles = WTiles(m_nr_tiles, m_tile_size);
  }
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
