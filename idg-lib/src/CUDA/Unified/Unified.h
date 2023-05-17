// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_UNIFIED_H_
#define IDG_CUDA_UNIFIED_H_

#include "CUDA/Generic/Generic.h"

namespace cu {
class UnifiedMemory;
}

namespace idg {
namespace proxy {
namespace cuda {

/**
 * @brief CUDA Proxy using Unified Memory
 *
 */
class Unified : public Generic {
 public:
  // Constructor
  Unified(ProxyInfo info = default_info());

  // Destructor
  ~Unified();

  void do_transform(DomainAtoDomainB direction) override;

  void set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) override;

  aocommon::xt::Span<std::complex<float>, 4>& get_final_grid() override;

  bool do_supports_wtiling() override { return false; }

 private:
  /**
   * Option to enable/disable reordering of the grid
   * to the host grid format, rather than the tiled
   * format used in the adder and splitter kernels.
   * Also set enable_tiling = false in
   * InstanceCUDA::launch_adder_unified and
   * InstanceCUDA::launch_splitter_unified to get
   * correct results.
   */
  bool m_enable_tiling = true;
  bool m_grid_is_tiled = false;

  // Override to return a pointer to its own 5D (tiled) grid,
  // to be used in Generic::run_imaging
  std::complex<float>* get_unified_grid_data() override {
    return unified_grid_tiled_.Span().data();
  }

  Tensor<std::complex<float>, 5> unified_grid_tiled_;
};  // class Unified

}  // namespace cuda
}  // namespace proxy
}  // namespace idg

#endif
