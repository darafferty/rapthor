#ifndef IDG_INIT_H_
#define IDG_INIT_H_

#include <iostream>
#include <complex>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "idg-util-config.h"
#include "idg-common.h" // idg data types
#include "idg-fft.h"

/* Macro */
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Constants */
#define SPEED_OF_LIGHT      299792458.0

/* Observation parameters */
#define START_FREQUENCY     150e6
#define FREQUENCY_INCREMENT 0.7e6
#define INTEGRATION_TIME    1.0f

namespace idg {

    /*
     * Memory-allocation is handled by Proxy
     */
    Array1D<float> get_example_frequencies(
        proxy::Proxy& proxy,
        unsigned int nr_channels,
        float start_frequency = START_FREQUENCY,
        float frequency_increment = FREQUENCY_INCREMENT);

    Array3D<Visibility<std::complex<float>>> get_dummy_visibilities(
        proxy::Proxy& proxy,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_channels);

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        proxy::Proxy& proxy,
        unsigned int nr_stations,
        unsigned int nr_baselines);

    Array2D<UVW<float>> get_example_uvw(
        proxy::Proxy& proxy,
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time = INTEGRATION_TIME);

    Array3D<std::complex<float>> get_zero_grid(
        proxy::Proxy& proxy,
        unsigned int nr_correlations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width);

    Array1D<unsigned int> get_example_aterms_offsets(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_timesteps);

    Array2D<float> get_example_spheroidal(
        proxy::Proxy& proxy,
        unsigned int height,
        unsigned int width);

    Array1D<float> get_zero_shift();

    /*
     * Default memory allocation
     */
    Array1D<float> get_example_frequencies(
        unsigned int nr_channels,
        float start_frequency = START_FREQUENCY,
        float frequency_increment = FREQUENCY_INCREMENT);

    Array3D<Visibility<std::complex<float>>> get_dummy_visibilities(
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_channels);

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        Array2D<UVW<float>> &uvw,
        Array1D<float> &frequencies,
        float        image_size,
        unsigned int grid_size,
        unsigned int nr_point_sources = 4,
        unsigned int max_pixel_offset = -1,
        unsigned int random_seed = 2,
        float        amplitude = 1);

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        unsigned int nr_stations,
        unsigned int nr_baselines);

    Array2D<UVW<float>> get_example_uvw(
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time = INTEGRATION_TIME);

    Array3D<std::complex<float>> get_zero_grid(
        unsigned int nr_correlations,
        unsigned int height,
        unsigned int width);

    Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
        unsigned int nr_timeslots,
        unsigned int nr_stations,
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

    void add_pt_src(
        Array3D<Visibility<std::complex<float>>> &visibilities,
        Array2D<UVW<float>> &uvw,
        Array1D<float> &frequencies,
        float          image_size,
        unsigned int   grid_size,
        float          x,
        float          y,
        float          amplitude);

    class Data {
        typedef struct { double x, y, z; } StationCoordinate;

        public:
            /*
             * Constructor
             */
            Data(
                unsigned int nr_stations_limit = 0, // infinite
                unsigned int baseline_length_limit = 0, // Meter
                std::string layout_file = "SKA1_low_ecef"
            );

            /*
             * Parameters
             */
            float compute_image_size(unsigned grid_size);
            float compute_grid_size(float image_size);
            static float compute_max_uv(unsigned grid_size, float image_size)
            {
                return (grid_size / image_size) / (SPEED_OF_LIGHT / start_frequency);
            }

            /*
             * Filter baselines
             */
            void filter_baselines(
                unsigned grid_size,
                float image_size);

            /*
             * Get methods
             */
            unsigned int get_nr_stations() const { return station_coordinates.size(); };

            unsigned int get_nr_baselines() const { return baselines.size(); };

            void get_frequencies(
                Array1D<float>& frequencies,
                float image_size,
                unsigned int channel_offset = 0) const;

            Array2D<UVW<float>> get_uvw(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int baseline_offset = 0,
                unsigned int time_offset = 0,
                float integration_time = INTEGRATION_TIME) const;

            void get_uvw(
                Array2D<UVW<float>>& uvw,
                unsigned int baseline_offset = 0,
                unsigned int time_offset = 0,
                float integration_time = INTEGRATION_TIME) const;

        private:
            /*
             * Set station_coordinates
             */
            void set_station_coordinates(
                std::string layout_file,
                unsigned nr_stations_limit = -1);

            std::vector<StationCoordinate> station_coordinates;

            /*
             * Set baselines and max_uv
             */
            void set_baselines(
                std::vector<StationCoordinate>& station_coordinates,
                unsigned int baseline_length_limit);

            std::vector<std::pair<float, Baseline>> baselines;
            float max_uv; // Meters

            /*
             * Helper methods
             */
            void evaluate_uvw(
                Baseline& baseline,
                unsigned int time,
                float integration_time,
                double* u, double* v, double* w) const;

            /*
             * Misc
             */
            const float pixel_padding = 0.8;

            /*
             * Observation parameters
             */
            static constexpr float start_frequency     = 150e6; // Mhz
            static constexpr float frequency_increment = 5e6; // Mhz
            static constexpr float observation_ra      = (10.0 * (M_PI/180.));
            static constexpr float observation_dec     = (70.0 * (M_PI/180.));
            static const int observation_year    = 2014;
            static const int observation_month   = 3;
            static const int observation_day     = 20;
            static const int observation_hour    = 1;
            static const int observation_minute  = 57;
            static const int observation_seconds = 0;
    };
} // namespace idg

#endif
