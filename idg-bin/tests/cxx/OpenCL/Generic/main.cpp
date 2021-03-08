// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-opencl.h"

#include "common.h"

int main(int argc, char *argv[]) {
  idg::proxy::opencl::Generic proxy;
  return compare_to_reference(proxy);
}
