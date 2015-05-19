#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "RW.h"

/*
	Library and source names
*/
#define SO_UTIL     "./Util.so"
#define SRC_UTIL    "Common/Util.cpp"
#define SRC_RW      "Common/RW.cpp"

/*
	Function names
*/
#define FUNCTION_WRITEUVGRID  "writeUVGrid"
#define FUNCTION_WRITEGRID    "writeGrid"
#define FUNCTION_WRITEVISIBILITIES "writeVisibilities"
std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize);

class Util {
    public:
        Util(
            const char *cc, const char *cflags,
            int nr_stations, int nr_baselines, int nr_time, int nr_channels,
            int nr_polarizations, int blocksize, int gridsize, float imagesize);
        void writeSubgrid(void *subgrid, const char *name);
        void writeGrid(void *grid, const char *name);
        void writeVisibilities(void *visibilities, const char *name);
    private:
        rw::Module *module;
};
