/*
    Structures
*/
typedef struct { float u, v, w; } UVW;
typedef struct { int x, y, z; } Coordinate;
typedef struct { int station1, station2; } Baseline;
typedef struct { int time_index; int nr_timesteps;
                 Baseline baseline; Coordinate coordinate;
                 Coordinate wtile_coordinate; int wtile_index;
                 int nr_aterms;} Metadata;

/*
    Index methods
*/
#define FUNCTION_ATTRIBUTES __device__
#include "common/Types.h"
