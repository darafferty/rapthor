// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "UnifiedOptimized.h"

namespace idg {
namespace proxy {
namespace hybrid {

// Constructor
UnifiedOptimized::UnifiedOptimized(ProxyInfo info) : Unified(info) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
#endif

  cpuProxy = new idg::proxy::cpu::Optimized();
}

// Destructor
UnifiedOptimized::~UnifiedOptimized() {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
#endif

  delete cpuProxy;
}

void UnifiedOptimized::do_transform(DomainAtoDomainB direction,
                                    Array3D<std::complex<float>>& grid) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
  std::cout << "Transform direction: " << direction << std::endl;
#endif

  cpuProxy->transform(direction, grid);
}  // end transform

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg

#include "UnifiedOptimizedC.h"
