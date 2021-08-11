// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Reference.h"
#include "ReferenceKernels.h"
#include "kernels/Kernels.h"

using namespace std;

namespace idg {
namespace proxy {
namespace cpu {

// Constructor
Reference::Reference() : CPU() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  m_kernels.reset(new kernel::cpu::ReferenceKernels());
}

}  // namespace cpu
}  // namespace proxy
}  // namespace idg
