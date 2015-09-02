#pragma once

/*
	Structures
*/
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y; } Coordinate;
typedef struct { int station1, station2; } Baseline;

/*
	Structure initializers
*/
inline UVW make_UVW(float u, float v, float w)
	{ UVW uvw = { u, v, w }; return uvw; };
inline Coordinate make_coordinate(int x, int y)
	{ Coordinate xy = { x, y }; return xy; };
inline Baseline make_baseline(int station1, int station2)
	{ Baseline bl = { station1, station2 }; return bl; };

/*
    Complex numbers
*/
#if defined __NVCC__
#define FLOAT_COMPLEX float2
#else
#define FLOAT_COMPLEX std::complex<float>
#endif

/*
    Derived parameters
*/
#if !defined NR_CHUNKS
#define NR_CHUNKS NR_TIME / CHUNKSIZE
#endif
#if !defined NR_BASELINES
#define NR_BASELINES (NR_STATIONS * (NR_STATIONS-1)) / 2
#endif

/*
	Datatypes
*/
typedef FLOAT_COMPLEX VisibilitiesType[NR_BASELINES][NR_TIME][NR_CHANNELS][NR_POLARIZATIONS];
typedef UVW UVWType[NR_BASELINES][NR_TIME];
typedef float WavenumberType[NR_CHANNELS];
typedef FLOAT_COMPLEX ATermType[NR_STATIONS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
typedef Baseline BaselineType[NR_BASELINES];
typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];

#if !defined ORDER
#define ORDER ORDER_BL_P_V_U
#endif

#define ORDER_BL_P_V_U 1
#define ORDER_BL_V_U_P 0

#if ORDER == ORDER_BL_V_U_P
typedef FLOAT_COMPLEX SubGridType[NR_BASELINES][NR_CHUNKS][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
#elif ORDER == ORDER_BL_P_V_U
typedef FLOAT_COMPLEX SubGridType[NR_BASELINES][NR_CHUNKS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
#endif


