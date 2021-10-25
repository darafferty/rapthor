// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_INDEX_H_
#define IDG_INDEX_H_

/* Index methods */
inline FUNCTION_ATTRIBUTES size_t index_grid_4d(int nr_polarizations,
                                                long grid_size, int i, int pol,
                                                int y, int x) {
  // grid: [*][nr_polarizations][grid_size][grid_size]
  return static_cast<size_t>(i) * nr_polarizations * grid_size * grid_size +
         static_cast<size_t>(pol) * grid_size * grid_size +
         static_cast<size_t>(y) * grid_size + static_cast<size_t>(x);
}

inline FUNCTION_ATTRIBUTES size_t index_grid_tiling(int nr_polarizations,
                                                    int tile_size,
                                                    size_t grid_size, int pol,
                                                    int y, int x) {
  // grid: [NR_TILES][NR_TILES][nr_polarizations][TILE_SIZE][TILE_SIZE]
  assert(grid_size % tile_size == 0);
  const int NR_TILES = grid_size / tile_size;
  size_t idx_tile_y = y / tile_size;
  size_t idx_tile_x = x / tile_size;
  size_t tile_y = y % tile_size;
  size_t tile_x = x % tile_size;

  return idx_tile_y * NR_TILES * nr_polarizations * tile_size * tile_size +
         idx_tile_x * nr_polarizations * tile_size * tile_size +
         pol * tile_size * tile_size + tile_y * tile_size + tile_x;
}

inline FUNCTION_ATTRIBUTES size_t index_grid_3d(size_t grid_size, int pol,
                                                int y, int x) {
  // grid: [nr_polarizations][grid_size][grid_size]
  return pol * grid_size * grid_size + y * grid_size + x;
}

inline FUNCTION_ATTRIBUTES size_t index_subgrid(int nr_polarizations,
                                                int subgrid_size, int s,
                                                int pol, int y, int x) {
  // subgrid: [nr_subgrids][nr_polarizations][subgrid_size][subgrid_size]
  return static_cast<size_t>(s) * nr_polarizations * subgrid_size *
             subgrid_size +
         static_cast<size_t>(pol) * subgrid_size * subgrid_size +
         static_cast<size_t>(y) * subgrid_size + static_cast<size_t>(x);
}

inline FUNCTION_ATTRIBUTES size_t index_visibility(int nr_correlations,
                                                   int nr_channels, int time,
                                                   int chan, int pol) {
  // visibilities: [nr_time][nr_channels][nr_correlations]
  return static_cast<size_t>(time) * nr_channels * nr_correlations +
         static_cast<size_t>(chan) * nr_correlations + static_cast<size_t>(pol);
}

inline FUNCTION_ATTRIBUTES size_t index_aterm(int subgrid_size,
                                              int nr_correlations,
                                              int nr_stations, int aterm_index,
                                              int station, int y, int x,
                                              int pol) {
  // aterm: [nr_aterms][subgrid_size][subgrid_size][nr_correlations]
  size_t aterm_nr = (aterm_index * nr_stations + station);
  return static_cast<size_t>(aterm_nr) * subgrid_size * subgrid_size *
             nr_correlations +
         static_cast<size_t>(y) * subgrid_size * nr_correlations +
         static_cast<size_t>(x) * nr_correlations + static_cast<size_t>(pol);
}

inline FUNCTION_ATTRIBUTES size_t
index_aterm_transposed(int nr_correlations, int subgrid_size, int nr_stations,
                       int aterm_index, int station, int y, int x, int pol) {
  // aterm: [nr_aterms][nr_correlations][subgrid_size][subgrid_size]
  size_t aterm_nr = (aterm_index * nr_stations + station);
  return static_cast<size_t>(aterm_nr) * nr_correlations * subgrid_size *
             subgrid_size +
         static_cast<size_t>(pol) * subgrid_size * subgrid_size +
         static_cast<size_t>(y) * subgrid_size + static_cast<size_t>(x);
}

#endif
