// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-cuda.h"

#include "../common/common.h"

int main(int argc, char *argv[]) {
  idg::proxy::cuda::Generic proxy;
  return compare_to_reference(proxy);
}
