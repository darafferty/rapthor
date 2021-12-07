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
  m_grid = grid;
  InstanceCUDA& device = get_device(0);
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Context& context = get_device(0).get_context();
  d_grid_.reset(new cu::DeviceMemory(context, grid->bytes()));
  device.copy_htod(htodstream, *d_grid_, grid->data(), grid->bytes());
  htodstream.synchronize();
}

std::shared_ptr<Grid> Generic::get_final_grid() {
  InstanceCUDA& device = get_device(0);
  cu::Stream& dtohstream = device.get_dtoh_stream();
  device.copy_dtoh(dtohstream, m_grid->data(), *d_grid_, m_grid->bytes());
  dtohstream.synchronize();
  return m_grid;
}

void Generic::check_grid() {
  if (!d_grid_) {
    throw std::runtime_error("device grid is not set, call set_grid first.");
  }
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
