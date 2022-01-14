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
  cu::Context& context = get_device(0).get_context();
  if (m_disable_wtiling) {
    InstanceCUDA& device = get_device(0);
    cu::Stream& htodstream = device.get_htod_stream();
    d_grid_.reset(new cu::DeviceMemory(context, grid->bytes()));
    htodstream.memcpyHtoD(*d_grid_, grid->data(), grid->bytes());
  }
  if (m_use_unified_memory) {
    cu::UnifiedMemory& u_grid = allocate_unified_grid(context, grid->bytes());
    char* first = reinterpret_cast<char*>(grid->data());
    char* result = reinterpret_cast<char*>(u_grid.data());
    std::copy_n(first, grid->bytes(), result);
  }
}

std::shared_ptr<Grid> Generic::get_final_grid() {
  if (!m_disable_wtiling) {
    flush_wtiles();
  }
  if (m_use_unified_memory) {
    cu::UnifiedMemory& u_grid = get_unified_grid();
    char* first = reinterpret_cast<char*>(u_grid.data());
    char* result = reinterpret_cast<char*>(m_grid->data());
    std::copy_n(first, m_grid->bytes(), result);
  } else if (m_disable_wtiling) {
    InstanceCUDA& device = get_device(0);
    cu::Stream& dtohstream = device.get_dtoh_stream();
    dtohstream.memcpyDtoH(m_grid->data(), *d_grid_, m_grid->bytes());
  }
  return m_grid;
}

std::unique_ptr<Plan> Generic::make_plan(
    const int kernel_size, const Array1D<float>& frequencies,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets, Plan::Options options) {
  if (do_supports_wtiling() && !m_disable_wtiling) {
    options.w_step = m_cache_state.w_step;
    options.nr_w_layers = std::numeric_limits<int>::max();
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
    init_buffers_wtiling(subgrid_size);
    m_wtiles = WTiles(m_nr_tiles, m_tile_size);
  }
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
