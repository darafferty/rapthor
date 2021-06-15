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
  // Defer call to cpuProxy
  // cpuProxy manages the wtiles state
  // plan will be made accordingly
  return cpuProxy->make_plan(kernel_size, frequencies, uvw, baselines,
                             aterms_offsets, options);
}

/*
 * FFT
 */
void GenericOptimized::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
  std::cout << "Transform direction: " << direction << std::endl;
#endif

  cpuProxy->transform(direction);
}  // end transform

/*
 * Grid
 */
void GenericOptimized::set_grid(std::shared_ptr<Grid> grid) {
  // Defer call to cpuProxy
  cpuProxy->set_grid(grid);
  CUDA::set_grid(grid);
}

std::shared_ptr<Grid> GenericOptimized::get_final_grid() {
  // Defer call to cpuProxy
  return cpuProxy->get_final_grid();
}

/*
 * Cache
 */
void GenericOptimized::init_cache(int subgrid_size, float cell_size,
                                  float w_step, const Array1D<float>& shift) {
  // Defer call to cpuProxy
  // cpuProxy manages the wtiles state
  cpuProxy->init_cache(subgrid_size, cell_size, w_step, shift);

  // Workaround for uninitialized m_cache_state in do_calibrate_init and
  // do_calibrate_update
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
