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
    Complex numbers
*/
#define FLOAT_COMPLEX fcomplex


/*
	Datatypes
*/
typedef UVW UVWType[1];
typedef FLOAT_COMPLEX VisibilitiesType[1][NR_POLARIZATIONS];
typedef float WavenumberType[1];
typedef FLOAT_COMPLEX ATermType[1][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
typedef FLOAT_COMPLEX GridType[1];
typedef FLOAT_COMPLEX SubGridType[1][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef Metadata MetadataType[1];

inline int index_subgrid(
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

inline int index_aterm(
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

inline int index_visibility(
    int nr_channels,
    int time,
    int chan)
{
    // visibilities: [nr_time][nr_channels][nr_polarizations]
    return time * nr_channels * NR_POLARIZATIONS +
           chan * NR_POLARIZATIONS;
}
