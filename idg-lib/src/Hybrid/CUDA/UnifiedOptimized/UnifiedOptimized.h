// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_HYBRID_UNIFIED_OPTIMIZED_H_
#define IDG_HYBRID_UNIFIED_OPTIMIZED_H_

#include "CUDA/Unified/Unified.h"
#include "CPU/Optimized/Optimized.h"

namespace idg {
namespace proxy {
namespace hybrid {

class UnifiedOptimized : public cuda::Unified {
 public:
  UnifiedOptimized(ProxyInfo info = default_info());

  ~UnifiedOptimized();

  virtual void do_transform(DomainAtoDomainB direction,
                            Array3D<std::complex<float>>& grid) override;

 private:
  idg::proxy::cpu::CPU* cpuProxy;

};  // class UnifiedOptimized

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg

#endif
