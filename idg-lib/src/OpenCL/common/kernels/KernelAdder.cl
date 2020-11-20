// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
    Kernel
*/
__kernel void kernel_adder(
    const int                grid_size,
    const int                subgrid_size,
    __global const Metadata* metadata,
    __global const float2*   subgrid,
    __global float2*         grid)
{
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int nr_threads = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Load position in grid
    const Metadata m = metadata[s];
    int grid_x = m.coordinate.x;
    int grid_y = m.coordinate.y;

    // Iterate all pixels in subgrid
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;
        float phase = M_PI*(x+y-subgrid_size)/subgrid_size;
        float2 phasor = (float2) (native_cos(phase), native_sin(phase));

        // Check wheter subgrid fits in grid
        if (grid_x >= 0 && grid_x < grid_size-subgrid_size &&
            grid_y >= 0 && grid_y < grid_size-subgrid_size) {
            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Add subgrid value to grid
            #pragma unroll 4
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int dst_idx = index_grid(grid_size, pol, grid_y + y, grid_x + x);
                int src_idx = index_subgrid(subgrid_size, s, pol, y_src, x_src);
                atomicAdd(&(grid[dst_idx]), cmul(phasor, subgrid[src_idx]));
            }
        }
    }
} // end kernel_adder
