#ifndef IDGWSCLEAN_TAPER_H_
#define IDGWSCLEAN_TAPER_H_

void init_optimal_taper_1D(int subgridsize, int padded_size, int size, float kernelsize, float* taper_subgrid, float* taper_grid);
void init_optimal_gridding_taper_1D(int subgridsize, int gridsize, float kernelsize, float* taper_subgrid, float* taper_grid);

#endif
