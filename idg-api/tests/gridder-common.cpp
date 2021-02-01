// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gridder-common.h"

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