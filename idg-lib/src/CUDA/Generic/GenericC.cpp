// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Generic.h"

extern "C" {
idg::proxy::Proxy* CUDA_Generic_create() {
  return new idg::proxy::cuda::Generic();
}
}  // end extern "C"
