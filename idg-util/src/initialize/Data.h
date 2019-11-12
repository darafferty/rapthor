#ifndef IDG_DATA_H_
#define IDG_DATA_H_

#include "idg-util-config.h"
#include "idg-common.h" // idg data types

namespace idg {

    class Data {
        typedef struct { double x, y, z; } StationCoordinate;

        public:
            /*
             * Constructor
             */
            Data(
                std::string layout_file = "SKA1_low_ecef"
            );

            /*
             * Parameters
             */
            float compute_image_size(unsigned long grid_size);
            float compute_max_uv(unsigned long grid_size);

            /*
             * Select baselines
             */
            // Maintain only the baselines up to max_uv meters
            void limit_max_baseline_length(float max_uv);

            // Maintain only n baselines, make sure to keep at least
            // a few long baselines (see fraction_long below)
            void limit_nr_baselines(unsigned int n);

            // Maintain only n stations, at random
            void limit_nr_stations(unsigned int n);

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

            float get_max_uv() const;

            /*
             * Misc methods
             */
            void print_info();

        private:
            /*
             * Set station_coordinates
             */
            void set_station_coordinates(
                std::string layout_file);

            std::vector<StationCoordinate> m_station_coordinates;

            /*
             * Set baselines
             */
            void set_baselines(
                std::vector<StationCoordinate>& station_coordinates);

            std::vector<std::pair<float, Baseline>> m_baselines;

            /*
             * Helper methods
             */
            void evaluate_uvw(
                Baseline& baseline,
                unsigned int timestep,
                float integration_time,
                double* u, double* v, double* w) const;

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

            /*
             * Imaging parameters
             */
            static constexpr float fov_deg       = 8.0;
            static constexpr float weight        = 1.0;
            static constexpr float grid_padding  = 1.25;
            static constexpr float fraction_long = 0.05; // fraction of long baselines to use
            static const int max_nr_stations     = 1000;
    };
} // namespace idg

#endif
