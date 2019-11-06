#ifndef IDG_DATA_H_
#define IDG_DATA_H_

#include "idg-util-config.h"
#include "idg-common.h" // idg data types

/* Constants */
#define SPEED_OF_LIGHT      299792458.0

namespace idg {

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
            unsigned int get_nr_stations() const { return m_station_coordinates.size(); };

            unsigned int get_nr_baselines() const { return m_baselines.size(); };

            void get_frequencies(
                Array1D<float>& frequencies,
                float image_size,
                unsigned int channel_offset = 0) const;

            Array2D<UVW<float>> get_uvw(
                unsigned int nr_baselines,
                unsigned int nr_timesteps,
                unsigned int baseline_offset = 0,
                unsigned int time_offset = 0,
                float integration_time = 1.0f) const;

            void get_uvw(
                Array2D<UVW<float>>& uvw,
                unsigned int baseline_offset = 0,
                unsigned int time_offset = 0,
                float integration_time = 1.0f) const;

            /*
             * Misc methods
             */
            void shuffle_stations();
            void print_info();

        private:
            /*
             * Set station_coordinates
             */
            void set_station_coordinates(
                std::string layout_file,
                unsigned nr_stations_limit = -1);

            std::vector<StationCoordinate> m_station_coordinates;

            /*
             * Set baselines and max_uv
             */
            void set_baselines(
                std::vector<StationCoordinate>& station_coordinates,
                unsigned int baseline_length_limit);

            std::vector<std::pair<float, Baseline>> m_baselines;
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

        public:
            /*
             * Observation parameters
             */
            static constexpr float start_frequency     = 150e6; // Mhz
            static constexpr float frequency_increment = 1e6; // Mhz
            static constexpr float observation_ra      = (10.0 * (M_PI/180.));
            static constexpr float observation_dec     = (70.0 * (M_PI/180.));
            static constexpr float integration_time    = 1.0;
            static const int observation_year    = 2014;
            static const int observation_month   = 3;
            static const int observation_day     = 20;
            static const int observation_hour    = 1;
            static const int observation_minute  = 57;
            static const int observation_seconds = 0;
    };
} // namespace idg

#endif
