// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

// This file contains common helper code for tDegridder and tGridder.

#include <idg-api.h>
#include <set>

enum class WMode { kNeither, kWStacking, kWTiling };

void AddWModeToOptions(WMode wmode, idg::api::options_type& options);

std::set<idg::api::Type> GetArchitectures();