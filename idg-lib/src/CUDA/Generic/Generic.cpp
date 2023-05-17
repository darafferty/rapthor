// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include "Generic.h"
#include "InstanceCUDA.h"
#include "kernels/KernelGridder.cuh"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {

// Constructor
Generic::Generic(ProxyInfo info) : CUDA(info) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  // This proxy supports three modes:
  //  1) Legacy: no W-Tiling, no Unified Memory
  //    This is the legacy mode, 'simple' GPU-only gridding/degridding.
  //  2) CUDA W-Tiling v1: W-Tiling, Unified Memory
  //    This is the first CUDA W-Tiling implementation. For simplicity,
  //    Unified Memory is used to go from tiles to the grid or vice versa.
  //  3) CUDA W-Tiling v2: W-Tiling, no Unified Memory
  //    This is the latest CUDA W-Tiling code, with explicit copies of patches
  //    between host and device for better use of PCIe bandwidth and therefore
  //    higher overall throughput.
  // All modes work correctly, but CUDA W-Tiling v2 is the best and therefore
  // the default setting. This mode is identicial to W-Tiling in
  // GenericOptimized.

  // Mode 1 (legacy)
  // m_disable_wtiling = true;
  // m_use_unified_memory = false;

  // Mode 2 (CUDA W-Tiling v1)
  // m_disable_wtiling = false;
  // m_use_unified_memory = true;

  // Mode 3 (CUDA W-Tiling v2)
  m_disable_wtiling = false;
  m_use_unified_memory = false;

  // There exists another legacy mode, without W-Tiling but with Unified Memory,
  // and another form of tiling. This mode is provided by the Unified proxy.
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
  const size_t nr_polarizations = grid.shape(1);
  const size_t grid_size = grid.shape(2);
  assert(grid.shape(3) == grid_size);
  const size_t sizeof_grid = grid.size() * sizeof(*grid.data());

  CUDA::set_grid(grid);
  cu::Context& context = get_device(0).get_context();
  if (m_disable_wtiling) {
    InstanceCUDA& device = get_device(0);
    cu::Stream& htodstream = device.get_htod_stream();
    d_grid_.reset(new cu::DeviceMemory(context, sizeof_grid));
    htodstream.memcpyHtoD(*d_grid_, grid.data(), sizeof_grid);
  }
  if (m_use_unified_memory) {
    Tensor<std::complex<float>, 3> unified_grid(
        std::make_unique<cu::UnifiedMemory>(context, sizeof_grid),
        {nr_polarizations, grid_size, grid_size});
    std::copy_n(grid.data(), grid.size(), get_unified_grid_data());
    set_unified_grid(std::move(unified_grid));
  }
}

aocommon::xt::Span<std::complex<float>, 4>& Generic::get_final_grid() {
  if (!m_disable_wtiling) {
    flush_wtiles();
  }

  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);

  if (m_use_unified_memory) {
    std::copy_n(get_unified_grid_data(), get_grid().size(), get_grid().data());
  } else if (m_disable_wtiling) {
    InstanceCUDA& device = get_device(0);
    cu::Stream& dtohstream = device.get_dtoh_stream();
    const size_t sizeof_grid = get_grid().size() * sizeof(*get_grid().data());
    dtohstream.memcpyDtoH(get_grid().data(), *d_grid_, sizeof_grid);
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
                  KernelGridder::block_size_x)
            : KernelGridder::block_size_x;
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
