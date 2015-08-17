#pragma once

/*
	Structures
*/
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y; } Coordinate;
typedef struct { int station1, station2; } Baseline;
typedef struct { int time_nr; Baseline baseline; Coordinate coordinate; } Metadata;

/*
    Complex numbers
*/
#define FLOAT_COMPLEX std::complex<float>

/*
	Datatypes
*/
typedef UVW UVWType[NR_SUBGRIDS][NR_TIMESTEPS];
typedef FLOAT_COMPLEX VisibilitiesType[NR_SUBGRIDS][NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
typedef float WavenumberType[NR_CHANNELS];
typedef FLOAT_COMPLEX ATermType[NR_STATIONS][NR_TIMESLOTS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];
typedef FLOAT_COMPLEX SubGridType[NR_BASELINES][NR_CHUNKS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef Metadata MetadataType[NR_SUBGRIDS];
