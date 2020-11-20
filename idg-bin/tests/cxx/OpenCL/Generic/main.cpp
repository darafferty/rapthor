// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-opencl.h"
#include "common.h"

using namespace std;

int main(int argc, char *argv[]) {
  // Compares to reference implementation
  int info = compare_to_reference<idg::proxy::opencl::Generic>();

  return info;
}
