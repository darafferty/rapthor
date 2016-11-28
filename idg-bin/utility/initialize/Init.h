#ifndef IDG_INIT_H_
#define IDG_INIT_H_

#include <iostream>
#include <complex>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "idg-config.h"
#include "idg-examples-config.h"
#include "idg-utility.h"
#include "idg-common.h" // idg data types
#include "idg-fft.h"

/* Macro */
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Constants */
#define RANDOM_SEED         1234
#define SPEED_OF_LIGHT      299792458.0

/* Observation parameters */
static const std::string ENV_LAYOUT_FILE;
#define LAYOUT_DIR          "src/utility/data"
#define LAYOUT_FILE         "SKA1_low_ecef"
#define START_FREQUENCY     150e6
#define FREQUENCY_INCREMENT 0.7e6
#define RIGHT_ASCENSION     (10.0 * (M_PI/180.))
#define DECLINATION         (70.0 * (M_PI/180.))
#define YEAR                2014
#define MONTH               03
#define DAY                 20
#define HOUR                01
#define MINUTE              57
#define SECONDS             1.3
#define INTEGRATION_TIME    1.0f

namespace idg {

/* Methods */

void init_zero_grid(
    void *ptr,
    int gridsize,
    int nr_polarizations);

void init_identity_aterm(
    void *ptr,
    int nr_timeslots,
    int nr_stations,
    int subgridsize,
    int nr_polarizations);

void init_identity_spheroidal(
    void *ptr,
    int subgridsize);

void init_zero_visibilities(
    void *ptr,
    int nr_baselines,
    int nr_time,
    int nr_channels,
    int nr_polarizations);

void init_example_uvw(
    void *ptr,
    int nr_stations,
    int nr_baselines,
    int nr_time,
    float integration_time = INTEGRATION_TIME);

void init_example_visibilities(
    void *ptr,
    int nr_baselines,
    int nr_time,
    int nr_channels,
    int nr_polarizations);

void add_pt_src(
    float x,
    float y,
    float amplitude,
    int nr_baselines,
    int nr_time,
    int nr_channels,
    int nr_polarizations,
    float imagesize,
    float gridsize,
    void *uvw,
    void *wavenumbers,
    void *visibilities);

void init_example_wavenumbers(
    void *ptr,
    int nr_channels);

void init_example_aterm(
    void *ptr,
    int nr_timeslots,
    int nr_stations,
    int subgridsize,
    int nr_polarizations);

void init_example_aterm_offsets(
    void *ptr,
    int nr_timeslots,
    int nr_time);

void init_example_spheroidal(
    void *ptr,
    int subgridsize);

void init_example_baselines(
    void *ptr,
    int nr_stations,
    int nr_baselines);

void init_example_grid(
    void *ptr,
    int gridsize,
    int nr_polarizations);

//////////////


void* init_zero_grid(
    int gridsize,
    int nr_polarizations);

void* init_identity_aterm(
    int nr_timeslots,
    int nr_stations,
    int subgridsize,
    int nr_polarizations);

void* init_zero_visibilities(
    int nr_baselines,
    int nr_time,
    int nr_channels,
    int nr_polarizations);

void* init_example_uvw(
    int nr_stations,
    int nr_baselines,
    int nr_time,
    float integration_time = INTEGRATION_TIME);

void* init_example_visibilities(
    int nr_baselines,
    int nr_time,
    int nr_channels,
    int nr_polarizations);

void* init_example_wavenumbers(int nr_channels);

void* init_example_aterm(
    int nr_timeslots,
    int nr_stations,
    int subgridsize,
    int nr_polarizations);

void* init_example_aterm_offsets(
    int nr_timeslots,
    int nr_time);

void* init_example_spheroidal(int subgridsize);

void* init_example_baselines(
    int nr_stations,
    int nr_baselines);

void* init_example_grid(
    int gridsize,
    int nr_polarizations);

} // namespace idg

#endif
