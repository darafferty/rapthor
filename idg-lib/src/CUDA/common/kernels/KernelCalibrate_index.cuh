inline __device__ long index_sums(
    unsigned int nr_polarizations,
    unsigned int total_nr_timesteps, // number of timesteps for all baselines
    unsigned int nr_channels,        // number channels for a single baseline
    unsigned int term_nr,
    unsigned int pol,
    unsigned int time,
    unsigned int chan)
{
    // sums: [MAX_NR_TERMS][NR_POLARIZATIONS][TOTAL_NR_TIMESTEPS][NR_CHANNELS]
    return term_nr * nr_polarizations * total_nr_timesteps * nr_channels +
           pol * total_nr_timesteps * nr_channels +
           time * nr_channels +
           chan;
}

inline __device__ long index_lmnp(
        unsigned int subgrid_size,
        unsigned int s,
        unsigned int y,
        unsigned int x)
{
    // lmnp: [NR_SUBGRIDS][SUBGRIDSIZE][SUBGRIDSIZE]
    return s * subgrid_size * subgrid_size +
           y * subgrid_size + x;
}
