#include <complex>

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
	Datatypes
*/
typedef FLOAT_COMPLEX VisibilitiesType[NR_BASELINES][NR_TIME][NR_CHANNELS][NR_POLARIZATIONS];
typedef UVW UVWType[NR_BASELINES][NR_TIME];
typedef UVW OffsetType[NR_BASELINES];
typedef float WavenumberType[NR_CHANNELS];
typedef FLOAT_COMPLEX ATermType[NR_STATIONS][NR_POLARIZATIONS][BLOCKSIZE][BLOCKSIZE];
typedef float SpheroidalType[BLOCKSIZE][BLOCKSIZE];
typedef Baseline BaselineType[NR_BASELINES];
typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];
typedef Coordinate CoordinateType[NR_BASELINES];

#if !defined ORDER
#define ORDER ORDER_BL_P_V_U
#endif

#define ORDER_BL_P_V_U 1
#define ORDER_BL_V_U_P 0

#if ORDER == ORDER_BL_V_U_P
typedef FLOAT_COMPLEX UVGridType[NR_BASELINES][BLOCKSIZE][BLOCKSIZE][NR_POLARIZATIONS];
#elif ORDER == ORDER_BL_P_V_U
typedef FLOAT_COMPLEX UVGridType[NR_BASELINES][NR_POLARIZATIONS][BLOCKSIZE][BLOCKSIZE];
#endif

