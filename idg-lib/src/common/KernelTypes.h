#define NR_CORRELATIONS 4

typedef struct { int x, y, z; } Coordinate;

typedef struct { unsigned int station1, station2; } Baseline;

typedef struct {
    int time_index;
    int nr_timesteps;
    int channel_begin;
    int channel_end;
    Baseline baseline;
    Coordinate coordinate;
    Coordinate wtile_coordinate;
    int wtile_index;
    int nr_aterms;
} Metadata;

template<class T>
struct UVW {T u; T v; T w;};
