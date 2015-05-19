#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>

#include "RW.h"

/*
    Library and source names
*/
#define SO_INIT     "./Init.so"
#define SRC_INIT    "Common/Init.cpp"
#define SRC_RW      "Common/RW.cpp"

/*
    Function names
*/
#define INIT_VISIBILITIES   "init_visibilities"
#define INIT_UVW            "init_uvw"
#define INIT_OFFSET         "init_offset"
#define INIT_WAVENUMBERS    "init_wavenumbers"
#define INIT_ATERM          "init_aterm"
#define INIT_SPHEROIDAL     "init_spheroidal"
#define INIT_BASELINES      "init_baselines"
#define INIT_COORDINATES    "init_coordinates"
#define INIT_UVGRID         "init_uvgrid"
#define INIT_GRID           "init_grid"

std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize);

class Init {
    public:
        Init(
            const char *cc, const char *cflags);
        void *init_visibilities();
        void *init_uvw();
        void *init_offset();
        void *init_wavenumbers();
        void *init_aterm();
        void *init_spheroidal();
        void *init_baselines();
        void *init_coordinates();
        void *init_uvgrid();
        void *init_grid();
        int get_nr_stations();
        int get_nr_baselines();
        int get_nr_baselines_data();
        int get_nr_time();
        int get_nr_time_data();
        int get_nr_channels();
        int get_nr_polarizations();
        int get_blocksize();
        int get_gridsize();
        float get_imagesize();
        
    private:
        rw::Module *module;
};
