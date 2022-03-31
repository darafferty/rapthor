// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-hybrid-cuda.h"

#include "../common/common.h"

int main(int argc, char* argv[]) {
  idg::proxy::cpu::Optimized proxy1;
  idg::proxy::hybrid::GenericOptimized proxy2;
  return compare(proxy1, proxy2);
}
