// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-cuda.h"

using ProxyType = idg::proxy::cuda::Unified;

#include "common.h"

int main(int argc, char **argv) {
  run();

  return 0;
}
