#pragma once

/*
	Structures
*/
#define TYPEDEF_UVW        typedef struct { float u, v, w; } UVW;
#define TYPEDEF_COORDINATE typedef struct { int x, y; } Coordinate;
#define TYPEDEF_BASELINE   typedef struct { int station1, station2; } Baseline;
#define TYPEDEF_METADATA   typedef struct { int baseline_offset; int time_offset; int nr_timesteps; \
                                            int aterm_index; Baseline baseline; Coordinate coordinate; } Metadata;


/*
    Complex numbers
*/
#define FLOAT_COMPLEX std::complex<float>

/*
	Datatypes
*/
#define TYPEDEF_UVW_TYPE          typedef UVW UVWType[1];
#define TYPEDEF_VISIBILITIES_TYPE typedef FLOAT_COMPLEX VisibilitiesType[1][nr_polarizations];
#define TYPEDEF_WAVENUMBER_TYPE   typedef float WavenumberType[nr_channels];
#define TYPEDEF_ATERM_TYPE        typedef FLOAT_COMPLEX ATermType[nr_stations][nr_timeslots][nr_polarizations][subgridsize][subgridsize];
#define TYPEDEF_SPHEROIDAL_TYPE   typedef float SpheroidalType[subgridsize][subgridsize];
#define TYPEDEF_GRID_TYPE         typedef FLOAT_COMPLEX GridType[nr_polarizations][gridsize][gridsize];
#define TYPEDEF_SUBGRID_TYPE      typedef FLOAT_COMPLEX SubGridType[1][nr_polarizations][subgridsize][subgridsize];
#define TYPEDEF_METADATA_TYPE     typedef Metadata MetadataType[1];
