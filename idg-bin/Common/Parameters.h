typedef struct {
    int nr_stations;
    int nr_baselines;
    int nr_time;
    int nr_channels;
    int nr_polarizations;
    int blocksize;
    int gridsize;
    float imagesize;
} Parameters;

Parameters parameters = {
NR_STATIONS, NR_BASELINES, NR_TIME, NR_CHANNELS,
NR_POLARIZATIONS, BLOCKSIZE, GRIDSIZE, IMAGESIZE
};

