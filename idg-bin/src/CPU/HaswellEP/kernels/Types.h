#pragma once

/*
    Structures
*/
// TODO: why are these here?
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y; } Coordinate;
typedef struct { int station1, station2; } Baseline;
typedef struct { int offset; int nr_timesteps;
                 Baseline baseline; Coordinate coordinate; } Metadata;

/*
    Complex numbers
*/
#define FLOAT_COMPLEX std::complex<float>

/*
    Datatypes
*/
// TODO: remove all of this stuff
typedef UVW UVWType[1];
typedef FLOAT_COMPLEX VisibilitiesType[1][NR_CHANNELS][NR_POLARIZATIONS];
typedef float WavenumberType[NR_CHANNELS];
typedef FLOAT_COMPLEX ATermType[NR_STATIONS][NR_TIMESLOTS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];
typedef FLOAT_COMPLEX SubGridType[1][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef Metadata MetadataType[1];
