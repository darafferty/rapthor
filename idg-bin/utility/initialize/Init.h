#ifndef IDG_INIT_H_
#define IDG_INIT_H_

#include <iostream>
#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "idg-config.h"
#include "idg-utility.h"

/* Macro */
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Constants */
#define RANDOM_SEED         1234
#define SPEED_OF_LIGHT      299792458.0

/* Observation parameters */
#define LAYOUT_DIR          "utility/initialize"
#define LAYOUT_FILE         "SKA1_low_ecef.txt"
#define START_FREQUENCY     500.0e6
#define FREQUENCY_INCREMENT 5.0e6
#define RIGHT_ASCENSION     (10.0 * (M_PI/180.))
#define DECLINATION         (70.0 * (M_PI/180.))
#define YEAR                2014
#define MONTH               03
#define DAY                 20
#define HOUR                01
#define MINUTE              57
#define SECONDS             1.3

namespace idg {

/* Methods */
void init_uvw(void *ptr, int nr_stations, int nr_baselines,
              int nr_time);
void init_visibilities(void *ptr, int nr_baselines, int nr_time,
                       int nr_channels, int nr_polarizations);
void init_wavenumbers(void *ptr, int nr_channels);
void init_aterm(void *ptr, int nr_stations, int nr_time,
                int nr_polarizations, int subgridsize);
void init_spheroidal(void *ptr, int subgridsize);
void init_baselines(void *ptr, int nr_stations, int nr_baselines);
void init_subgrid(void *ptr, int nr_baselines, int subgridsize,
                  int nr_polarizations, int nr_chunks);
void init_grid(void *ptr, int gridsize, int nr_polarizations);
void init_metadata(void *ptr, void *uvw, void *wavenumbers,
                   int nr_stations, int nr_baselines, int nr_timesteps,
                   int nr_timeslots, int nr_channels, int gridsize,
                   int subgridsize, float imagesize);

void* init_uvw(int nr_stations, int nr_baselines, int nr_time);
void* init_visibilities(int nr_baselines, int nr_time, int nr_channels,
                        int nr_polarizations);
void* init_wavenumbers(int nr_channels);
void* init_aterm(int nr_stations, int nr_time, int nr_polarizations,
                 int subgridsize);
void* init_spheroidal(int subgridsize);
void* init_baselines(int nr_stations, int nr_baselines);
void* init_subgrid(int nr_baselines, int subgridsize, int nr_polarizations,
                   int nr_chunks);
void* init_grid(int gridsize, int nr_polarizations);
void* init_metadata(void *uvw, void *wavenumbers, int nr_stations,
                    int nr_baselines, int nr_timesteps, int nr_timeslots,
                    int nr_channels, int gridsize, int subgridsize,
                    float imagesize);

} // namespace idg

#endif
