// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
    Kernel
*/
__kernel void kernel_scaler(
    const int        subgrid_size,
    __global float2* subgrid)
{
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Compute scaling factor
    const float scale = 1.0 / ((float) subgrid_size * (float) subgrid_size);

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += blocksize) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;

        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            int idx = index_subgrid(subgrid_size, s, pol, y, x);
            float2 value = subgrid[idx];
            subgrid[idx] = (float2) (value.x * scale, value.y * scale);
        }
    }
}
