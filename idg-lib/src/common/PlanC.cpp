// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Plan.h"

extern "C" {

int Plan_get_nr_subgrids(std::unique_ptr<idg::Plan>* plan) {
  return (*plan)->get_nr_subgrids();
}

void Plan_copy_metadata(std::unique_ptr<idg::Plan>* plan, void* ptr) {
  (*plan)->copy_metadata(ptr);
}

void Plan_destroy(std::unique_ptr<idg::Plan>* plan) { delete plan; }

}  // extern "C"
