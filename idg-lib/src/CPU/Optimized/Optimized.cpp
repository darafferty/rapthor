// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Optimized.h"
#include "OptimizedKernels.h"

using namespace std;

namespace idg {
namespace proxy {
namespace cpu {

// Constructor
Optimized::Optimized() : CPU() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  m_kernels.reset(new kernel::cpu::OptimizedKernels());
}

}  // namespace cpu
}  // namespace proxy
}  // namespace idg
