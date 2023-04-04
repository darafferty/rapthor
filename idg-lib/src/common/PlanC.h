// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

int Plan_get_nr_subgrids(struct Plan* plan);
void Plan_copy_metadata(struct Plan* plan, void* ptr);
void Plan_copy_aterm_indices(struct Plan* plan, void* ptr);
void Plan_destroy(struct Plan* plan);
struct Plan* Plan_init(struct Plan* plan);