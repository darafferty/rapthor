// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-cuda.h"

using ProxyType = idg::proxy::cuda::Generic;

#include "../common/common.h"

int main(int argc, char* argv[]) {
  run(argc, argv);

  return 0;
}
