#pragma once

/*
    Structures
*/
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y, z; } Coordinate;
typedef struct { int station1, station2; } Baseline;
typedef struct { int baseline_offset; int time_offset; int nr_timesteps;
                 int aterm_index;
                 Baseline baseline; Coordinate coordinate; } Metadata;

/*
    Index methods
*/

inline __device__ long index_grid(
        long grid_size,
        int pol,
        int y,
        int x)
{
    // grid: [NR_POLARIZATIONS][grid_size][grid_size]
    return pol * grid_size * grid_size +
           y * grid_size +
           x;
}

inline __device__ long index_grid_tiled(
        long grid_size,
        int pol,
        int y,
        int x)
{
    // grid: [NR_TILES][NR_TILES][NR_POLARIZATIONS][TILE_SIZE][TILE_SIZE]
    const int TILE_SIZE = TILE_SIZE_GRID;
    const int NR_TILES  = grid_size / TILE_SIZE;
    long idx_tile_y = y / TILE_SIZE;
    long idx_tile_x = x / TILE_SIZE;
    long tile_y = y % TILE_SIZE;
    long tile_x = x % TILE_SIZE;

    return idx_tile_y * NR_TILES * NR_POLARIZATIONS * TILE_SIZE * TILE_SIZE +
           idx_tile_x * NR_POLARIZATIONS * TILE_SIZE * TILE_SIZE +
           pol * TILE_SIZE * TILE_SIZE +
           tile_y * TILE_SIZE +
           tile_x;
}

inline __device__ long index_subgrid(
    int subgrid_size, 
    int s,
    int pol,
    int y,
    int x)
{
    // subgrid: [nr_subgrids][NR_POLARIZATIONS][subgrid_size][subgrid_size]
   return s * NR_POLARIZATIONS * subgrid_size * subgrid_size +
          pol * subgrid_size * subgrid_size +
          y * subgrid_size +
          x;
}

inline __device__ int index_aterm(
    int subgrid_size,
    int nr_stations,
    int aterm_index,
    int station,
    int y,
    int x)
{
    // aterm: [nr_aterms][subgrid_size][subgrid_size][NR_POLARIZATIONS]
    int aterm_nr = (aterm_index * nr_stations + station);
    return aterm_nr * subgrid_size * subgrid_size * NR_POLARIZATIONS +
           y * subgrid_size * NR_POLARIZATIONS +
           x * NR_POLARIZATIONS;
}

inline __device__ int index_visibility(
    int nr_channels,
    int time,
    int chan,
    int pol)
{
    // visibilities: [nr_time][nr_channels][nr_polarizations]
    return time * nr_channels * NR_POLARIZATIONS +
           chan * NR_POLARIZATIONS +
           pol;
}
