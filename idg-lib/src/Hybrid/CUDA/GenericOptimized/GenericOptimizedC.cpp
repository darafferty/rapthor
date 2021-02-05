// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "GenericOptimized.h"

extern "C" {
idg::proxy::Proxy* HybridCUDA_GenericOptimized_create() {
  return new idg::proxy::hybrid::GenericOptimized();
}
}
