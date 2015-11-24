#ifndef IDG_ARGS_H_
#define IDG_ARGS_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
 
#include "optionparser.h"
 
namespace idg {

/* Parameters */
enum  optionIndex { UNKNOWN, NR_STATIONS, NR_TIME, NR_CHANNELS, W_PLANES, GRIDSIZE, SUBGRIDSIZE, CHUNKSIZE, JOBSIZE, DEVICE, NR_STREAMS };

const option::Descriptor usage[] = {
    {UNKNOWN,     0, "", ""    , option::Arg::None, "Parameters:" },
    {NR_STATIONS, 0, "", "nr_stations", option::Arg::Optional, "  --nr_stations  \tNumber of stations."},
    {NR_TIME,     0, "", "nr_time",     option::Arg::Optional, "  --nr_time      \tNumber of timesteps."},
    {NR_CHANNELS, 0, "", "nr_channels", option::Arg::Optional, "  --nr_channels  \tNumber of channels."},
    {W_PLANES,    0, "", "w_planes",    option::Arg::Optional, "  --w_planes     \tNumber of w planes."},
    {GRIDSIZE,    0, "", "gridsize",    option::Arg::Optional, "  --gridsize     \tSize of grid/image"},
    {SUBGRIDSIZE, 0, "", "subgridsize", option::Arg::Optional, "  --subgridsize  \tSize of subgrids."},
    {CHUNKSIZE,   0, "", "chunksize",   option::Arg::Optional, "  --chunksize    \tNumber of samples per subgrid."},
    {JOBSIZE,     0, "", "jobsize",     option::Arg::Optional, "  --jobsize      \tAmount of work in parallel."},
    {0,0,0,0,0,0}
};

const option::Descriptor usage2[] = {
    {UNKNOWN,     0, "", ""    , option::Arg::None, "Parameters:" },
    {NR_STATIONS, 0, "", "nr_stations", option::Arg::Optional, "  --nr_stations  \tNumber of stations."},
    {NR_TIME,     0, "", "nr_time",     option::Arg::Optional, "  --nr_time      \tNumber of timesteps."},
    {NR_CHANNELS, 0, "", "nr_channels", option::Arg::Optional, "  --nr_channels  \tNumber of channels."},
    {W_PLANES,    0, "", "w_planes",    option::Arg::Optional, "  --w_planes     \tNumber of w planes."},
    {GRIDSIZE,    0, "", "gridsize",    option::Arg::Optional, "  --gridsize     \tSize of grid/image"},
    {SUBGRIDSIZE, 0, "", "subgridsize", option::Arg::Optional, "  --subgridsize  \tSize of subgrids."},
    {CHUNKSIZE,   0, "", "chunksize",   option::Arg::Optional, "  --chunksize    \tNumber of samples per subgrid."},
    {JOBSIZE,     0, "", "jobsize",     option::Arg::Optional, "  --jobsize      \tAmount of work in parallel."},
    {DEVICE,      0, "", "device",      option::Arg::Optional, "  --device       \tNumber of device to use."},
    {NR_STREAMS,  0, "", "nr_streams",  option::Arg::Optional, "  --nr_streams   \tNumber of parallel streams."},
    {0,0,0,0,0,0}
};


inline int read_int(const char *string) {
    return !string || strlen(string) < 1 ? 0 : atoi(string);
}


void get_parameters(
    int argc, char **argv,
    int *nr_stations, int *nr_time,
    int *nr_channels, int *w_planes,
    int *gridsize,    int *subgridsize,
    int *chunksize,   int *jobsize) {

    // Initialize argument parser
    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
    option::Stats  stats(usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage, argc, argv, options, buffer);

    // Check if parsing succeeded
    if (parse.error()) {
        fprintf(stderr, "Error parsing arguments\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arguments
    *nr_time       = -1;
    *nr_channels   = -1;
    *w_planes      = -1;
    *gridsize      = -1;
    *subgridsize   = -1;
    *chunksize     = -1;
    *jobsize       = -1;
    
    // Try to read all arguments
    if (options[NR_STATIONS]) *nr_stations = read_int(options[NR_STATIONS].arg);
    if (options[NR_TIME])     *nr_time     = read_int(options[NR_TIME].arg);
    if (options[NR_CHANNELS]) *nr_channels = read_int(options[NR_CHANNELS].arg);
    if (options[W_PLANES])    *w_planes    = read_int(options[W_PLANES].arg);
    if (options[GRIDSIZE])    *gridsize    = read_int(options[GRIDSIZE].arg);
    if (options[SUBGRIDSIZE]) *subgridsize = read_int(options[SUBGRIDSIZE].arg);
    if (options[CHUNKSIZE])   *chunksize   = read_int(options[CHUNKSIZE].arg);
    if (options[JOBSIZE])     *jobsize     = read_int(options[JOBSIZE].arg);

    if ((*nr_stations) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*nr_time) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*nr_channels) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*w_planes) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*gridsize) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*subgridsize) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*chunksize) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
    if ((*jobsize) == -1) {
        option::printUsage(std::cerr, usage);
        exit(EXIT_FAILURE);
    }
}

void get_parameters(
    int argc, char **argv,
    int *nr_stations, int *nr_time,
    int *nr_channels, int *w_planes,
    int *gridsize,    int *subgridsize,
    int *chunksize,   int *jobsize,
    int *device_number, int *nr_streams) {

    // Initialize argument parser
    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
    option::Stats  stats(usage2, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage2, argc, argv, options, buffer);

    // Check if parsing succeeded
    if (parse.error()) {
        fprintf(stderr, "Error parsing arguments\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arguments
    *nr_stations   = -1;
    *nr_time       = -1;
    *nr_channels   = -1;
    *w_planes      = -1;
    *gridsize      = -1;
    *subgridsize   = -1;
    *chunksize     = -1;
    *jobsize       = -1;
    *device_number = -1;
    *nr_streams    = -1;
    
    // Try to read all arguments
    if (options[NR_STATIONS]) *nr_stations   = read_int(options[NR_STATIONS].arg);
    if (options[NR_TIME])     *nr_time       = read_int(options[NR_TIME].arg);
    if (options[NR_CHANNELS]) *nr_channels   = read_int(options[NR_CHANNELS].arg);
    if (options[W_PLANES])    *w_planes      = read_int(options[W_PLANES].arg);
    if (options[GRIDSIZE])    *gridsize      = read_int(options[GRIDSIZE].arg);
    if (options[SUBGRIDSIZE]) *subgridsize   = read_int(options[SUBGRIDSIZE].arg);
    if (options[CHUNKSIZE])   *chunksize     = read_int(options[CHUNKSIZE].arg);
    if (options[JOBSIZE])     *jobsize       = read_int(options[JOBSIZE].arg);
    if (options[DEVICE])      *device_number = read_int(options[DEVICE].arg);
    if (options[NR_STREAMS])  *nr_streams    = read_int(options[NR_STREAMS].arg);

    // Check all arguments
    if (
       ((*nr_stations)   == -1) ||
       ((*nr_time)       == -1) ||
       ((*nr_channels)   == -1) ||
       ((*w_planes)      == -1) ||
       ((*gridsize)      == -1) ||
       ((*subgridsize)   == -1) ||
       ((*chunksize)     == -1) ||
       ((*jobsize)       == -1) ||
       ((*device_number) == -1) ||
       ((*nr_streams)    == -1))
       {
        option::printUsage(std::cerr, usage2);
        exit(EXIT_FAILURE);
    }
}

} // namespace idg

#endif
