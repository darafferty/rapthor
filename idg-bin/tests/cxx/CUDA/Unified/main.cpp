// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "idg-cuda.h"

using namespace std;

using ProxyType = idg::proxy::cuda::Unified;

#include "../common/common.h"

int main(int argc, char *argv[]) { return compare_to_reference(); }
