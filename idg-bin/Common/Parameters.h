typedef struct {
    int nr_stations;
    int nr_time;
    int nr_channels;
    int nr_polarizations;
    int subgridsize;
    int gridsize;
    int chunksize;
    float imagesize;
} Parameters;

Parameters parameters = {
NR_STATIONS, NR_TIME, NR_CHANNELS,
NR_POLARIZATIONS, SUBGRIDSIZE, GRIDSIZE,
CHUNKSIZE, IMAGESIZE
};

