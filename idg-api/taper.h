// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDGWSCLEAN_TAPER_H_
#define IDGWSCLEAN_TAPER_H_

void init_optimal_taper_1D(int subgridsize, int padded_size, int size,
                           float kernelsize, float* taper_subgrid,
                           float* taper_grid);
void init_optimal_gridding_taper_1D(int subgridsize, int gridsize,
                                    float kernelsize, float* taper_subgrid,
                                    float* taper_grid);

void init_kaiser_bessel_1D(int size, float* taper_grid);

void init_blackman_harris_1D(int size, float* taper_grid);

#endif
