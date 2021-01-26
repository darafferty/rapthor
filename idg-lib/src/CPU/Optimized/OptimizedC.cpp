// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Optimized.h"

extern "C" {
struct idg::proxy::Proxy* CPU_Optimized_create() {
  return new idg::proxy::cpu::Optimized();
}
}
