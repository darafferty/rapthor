// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Reference.h"

extern "C" {
idg::proxy::Proxy* CPU_Reference_create() {
  return new idg::proxy::cpu::Reference();
}
}  // end extern "C"
