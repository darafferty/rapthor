typedef struct { int x, y, z; } Coordinate;

typedef struct { unsigned int station1, station2; } Baseline;

typedef struct {
    int time_index;
    int nr_timesteps;
    Baseline baseline;
    Coordinate coordinate;
    Coordinate wtile_coordinate;
    int wtile_index;
    int nr_aterms;
} Metadata;

template<class T>
struct Matrix2x2 {T xx; T xy; T yx; T yy;};

template<class T>
using Visibility = Matrix2x2<T>;

template<class T>
struct UVW {T u; T v; T w;};
