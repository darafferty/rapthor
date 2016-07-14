#pragma once

/*
	Structures
*/
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y; } Coordinate;
typedef struct { int station1, station2; } Baseline;
typedef struct { int baseline_offset; int time_offset; int nr_timesteps;
                 int aterm_index; Baseline baseline; Coordinate coordinate; } Metadata;

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
typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];
typedef FLOAT_COMPLEX SubGridType[1][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef Metadata MetadataType[1];
