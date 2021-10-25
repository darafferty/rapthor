// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
	Index methods
*/
inline int index_grid(
        int grid_size,
        int pol,
        int y,
        int x)
{
    // grid: [NR_CORRELATIONS][grid_size][grid_size]
    return pol * grid_size * grid_size +
           y * grid_size +
           x;
}

inline int index_subgrid(
    int subgrid_size,
    int s,
    int pol,
    int y,
    int x)
{
    // subgrid: [nr_subgrids][NR_CORRELATIONS][subgrid_size][subgrid_size]
   return s * NR_CORRELATIONS * subgrid_size * subgrid_size +
          pol * subgrid_size * subgrid_size +
          y * subgrid_size +
          x;
}

inline int index_aterm(
    int subgrid_size,
    int nr_stations,
    int aterm_index,
    int station,
    int y,
    int x)
{
    // aterm: [nr_aterms][subgrid_size][subgrid_size][NR_CORRELATIONS]
    int aterm_nr = (aterm_index * nr_stations + station);
    return aterm_nr * subgrid_size * subgrid_size * NR_CORRELATIONS +
           y * subgrid_size * NR_CORRELATIONS +
           x * NR_CORRELATIONS;
}

inline int index_visibility(
    int nr_channels,
    int time,
    int chan)
{
    // visibilities: [nr_time][nr_channels][nr_correlations]
    return time * nr_channels * NR_CORRELATIONS +
           chan * NR_CORRELATIONS;
}
