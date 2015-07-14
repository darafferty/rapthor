/*
    Derived parameters
*/
#define NR_CHUNKS NR_TIME / CHUNKSIZE


/*
    Structures
*/
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y; } Coordinate;
typedef struct { int station1, station2; } Baseline;


/*
    Datatypes
*/
typedef fcomplex VisibilitiesType[JOBSIZE][NR_TIME][NR_CHANNELS][NR_POLARIZATIONS];
typedef UVW UVWType[JOBSIZE][NR_TIME];
typedef float WavenumberType[NR_CHANNELS];
typedef fcomplex ATermType[NR_STATIONS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
typedef Baseline BaselineType[JOBSIZE];
typedef fcomplex GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];

#define ORDER_BL_P_V_U 1
#define ORDER_BL_V_U_P 0
#if ORDER == ORDER_BL_V_U_P
typedef fcomplex SubGridType[JOBSIZE][NR_CHUNKS][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
#elif ORDER == ORDER_BL_P_V_U
typedef fcomplex SubGridType[JOBSIZE][NR_CHUNKS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
#endif


