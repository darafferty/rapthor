// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gridder-common.h"

#include "idg-config.h"

void AddWModeToOptions(WMode wmode, idg::api::options_type& options) {
  switch (wmode) {
    case WMode::kNeither:
      options["disable_wtiling"] = true;
      options["disable_wstacking"] = true;
      break;
    case WMode::kWStacking:
      options["disable_wtiling"] = true;
      break;
    case WMode::kWTiling:
      // Do not disable anything.
      break;
  }
}

std::set<idg::api::Type> GetArchitectures() {
  std::set<idg::api::Type> architectures;
#if defined(BUILD_LIB_CPU)
  architectures.insert(idg::api::Type::CPU_REFERENCE);
  architectures.insert(idg::api::Type::CPU_OPTIMIZED);
#endif
#if defined(BUILD_LIB_CUDA)
  architectures.insert(idg::api::Type::CUDA_GENERIC);
#endif
#if defined(BUILD_LIB_CPU) && defined(BUILD_LIB_CUDA)
  architectures.insert(idg::api::Type::HYBRID_CUDA_CPU_OPTIMIZED);
#endif
  return architectures;
}