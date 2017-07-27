#ifndef IDG_INIT_H_
#define IDG_INIT_H_

#include <iostream>
#include <complex>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "idg-config.h"
#include "idg-examples-config.h"
#include "idg-common.h" // idg data types
#include "idg-fft.h"

/* Macro */
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Constants */
#define RANDOM_SEED         1234
#define SPEED_OF_LIGHT      299792458.0

/* Observation parameters */
#define LAYOUT_DIR          "src/utility/data"
#define LAYOUT_FILE         "SKA1_low_ecef"
#define ENV_LAYOUT_FILE     "LAYOUT_FILE"
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

    Array1D<float> get_example_frequencies(
        unsigned int nr_channels,
        float start_frequency = START_FREQUENCY,
        float frequency_increment = FREQUENCY_INCREMENT);

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        unsigned int nr_stations,
        unsigned int nr_timesteps,
        unsigned int nr_channels);

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        unsigned int nr_stations,
        unsigned int nr_baselines);

    Array2D<UVWCoordinate<float>> get_example_uvw(
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time = INTEGRATION_TIME);

    Array3D<std::complex<float>> get_zero_grid(
        unsigned int nr_correlations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array1D<unsigned int> get_example_aterms_offsets(
        unsigned int nr_timeslots,
        unsigned int nr_timesteps);

    Array2D<float> get_identity_spheroidal(
        unsigned int height,
        unsigned int width);

    Array2D<float> get_example_spheroidal(
        unsigned int height,
        unsigned int width);

    float evaluate_spheroidal(float nu);

} // namespace idg

#endif
