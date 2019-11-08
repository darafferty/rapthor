#include <algorithm>

#include "uvwsim.h"

#include "Data.h"

namespace idg{

    Data::Data(
        std::string layout_file)
    {
        // Set station_coordinates
        set_station_coordinates(layout_file);

        // Shuffle stations
        shuffle_stations();

        // Set baselines
        set_baselines(m_station_coordinates);
    }

    void Data::set_station_coordinates(
        std::string layout_file = "SKA1_low_ecef")
    {
        // Check whether layout file exists
        std::string filename = std::string(IDG_DATA_DIR) + "/" + layout_file;
        if (!uvwsim_file_exists(filename.c_str())) {
            std::cerr << "Failed to find specified layout file: "
                      << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        // Get number of stations
        unsigned nr_stations = uvwsim_get_num_stations(filename.c_str());
        if (nr_stations < 0) {
            std::cerr << "Failed to read any stations from layout file" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Allocate memory for station coordinates
        double* x = (double *) malloc(nr_stations * sizeof(double));
        double* y = (double *) malloc(nr_stations * sizeof(double));
        double* z = (double *) malloc(nr_stations * sizeof(double));

        // Load the antenna coordinates
        unsigned nr_stations_read = uvwsim_load_station_coords(filename.c_str(), nr_stations, x, y, z);
        if (nr_stations_read != nr_stations) {
            std::cerr << "Failed to read antenna coordinates." << std::endl;
            exit(EXIT_FAILURE);
        }

        // Create vector of station coordinates
        for (unsigned i = 0; i < nr_stations; i++) {
            Data::StationCoordinate coordinate = { x[i], y[i], z[i] };
            m_station_coordinates.push_back(coordinate);
        }

        // Free memory
        free(x); free(y); free(z);
    }

    void Data::shuffle_stations() {
        auto nr_stations = m_station_coordinates.size();

        // Pick random station indices
        std::srand(0);
        std::vector<unsigned> station_indices(nr_stations);
        for (unsigned i = 0; i < nr_stations; i++) {
            station_indices[i] = i;
        }
        std::random_shuffle(station_indices.begin(), station_indices.end());

        std::vector<StationCoordinate> station_coordinates_shuffled;

        for (auto i : station_indices) {
            station_coordinates_shuffled.push_back(m_station_coordinates[i]);
        }

        std::swap(m_station_coordinates, station_coordinates_shuffled);
    }

    void Data::print_info() {
        std::cout << "number of stations: "
                  << m_station_coordinates.size() << std::endl;
        std::cout << "number of baselines: "
                  << m_baselines.size() << std::endl;
        std::cout << "longest baseline = " << get_max_uv() << std::endl;
    }

    float Data::compute_image_size(unsigned grid_size)
    {
        auto max_uv = get_max_uv();
        return grid_size / max_uv * (SPEED_OF_LIGHT / start_frequency);
    }

    float Data::compute_grid_size(float image_size)
    {
        auto max_uv = get_max_uv();
        return (image_size * pixel_padding) / (SPEED_OF_LIGHT / start_frequency) * max_uv;
    };

    void Data::set_baselines(
        std::vector<StationCoordinate>& station_coordinates)
    {
        // Compute (maximum) baseline length and select baselines
        unsigned nr_stations = station_coordinates.size();

        for (unsigned station1 = 0; station1 < nr_stations; station1++) {
            for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
                Baseline baseline = {station1, station2};
                double u, v, w;

                // Compute uvw values for 12 hours of observation (with steps of 1 hours)
                float max_uv = 0.0f;
                for (unsigned time = 0; time < 12; time++) {
                    evaluate_uvw(baseline, time, 3600, &u, &v, &w);
                    float baseline_length = sqrtf(u*u + v*v);
                    max_uv = std::max(max_uv, baseline_length);
                } // end for time

                // Add baseline
                m_baselines.push_back(std::pair<float, Baseline>(max_uv, baseline));
            } // end for station 2
        } // end for station 1
    }

    void Data::filter_baselines(
        unsigned grid_size,
        float image_size)
    {
        #if defined(DEBUG)
        std::cout << "Data::" << __func__ << std::endl;
        std::cout << "grid_size = " << grid_size << ", image_size = " << image_size << std::endl;
        #endif

        // Select the baselnes that fit in the grid,
        // keep track of stations corresponding to these baselines
        bool selected_baselines[m_baselines.size()];
        bool selected_stations[m_station_coordinates.size()];
        memset(selected_baselines, 0, sizeof(selected_baselines));

        #pragma omp parallel for
        for (unsigned i = 0; i < m_baselines.size(); i++) {
            std::pair<float, Baseline> entry = m_baselines[i];
            float baseline_length_meters = entry.first;
            float baseline_length_pixels = baseline_length_meters * image_size * (start_frequency / SPEED_OF_LIGHT);
            selected_baselines[i] = baseline_length_pixels < grid_size;
        }

        // Put all selected baselines into a new vector
        std::vector<std::pair<float, Baseline>> baselines_;

        for (unsigned i = 0; i < m_baselines.size(); i++) {
            if (selected_baselines[i]) {
                baselines_.push_back(m_baselines[i]);
                Baseline bl = m_baselines[i].second;
                selected_stations[bl.station1] = true;
                selected_stations[bl.station2] = true;
            }
        }

        // Put all selected stations into a new vector
        std::vector<StationCoordinate> station_coordinates_;

        for (unsigned i = 0; i < m_station_coordinates.size(); i++) {
            if (selected_stations[i]) {
                station_coordinates_.push_back(m_station_coordinates[i]);
            }
        }

        // Replace members
        std::swap(m_baselines, baselines_);
        std::swap(m_station_coordinates, station_coordinates_);
    }

    void Data::limit_nr_baselines(
        unsigned int n)
    {
        #if defined(DEBUG)
        std::cout << "Data::" << __func__ << std::endl;
        #endif

        // Select the first n baselnes,
        // keep track of stations corresponding to these baselines
        bool selected_stations[m_station_coordinates.size()];

        // Put the first n baselines into a new vector
        std::vector<std::pair<float, Baseline>> baselines_;

        for (unsigned i = 0; i <n; i++) {
            baselines_.push_back(m_baselines[i]);
            Baseline bl = m_baselines[i].second;
            selected_stations[bl.station1] = true;
            selected_stations[bl.station2] = true;
        }

        // Put all selected stations into a new vector
        std::vector<StationCoordinate> station_coordinates_;

        for (unsigned i = 0; i < m_station_coordinates.size(); i++) {
            if (selected_stations[i]) {
                station_coordinates_.push_back(m_station_coordinates[i]);
            }
        }

        // Replace members
        std::swap(m_baselines, baselines_);
        std::swap(m_station_coordinates, station_coordinates_);
    }

    void Data::get_frequencies(
        Array1D<float>& frequencies,
        float image_size,
        unsigned int channel_offset) const
    {
        auto nr_channels = frequencies.get_x_dim();
        auto max_uv = get_max_uv();
        float frequency_increment = SPEED_OF_LIGHT / (max_uv * image_size);
        for (unsigned chan = 0; chan < nr_channels; chan++) {
            frequencies(chan) = start_frequency + frequency_increment * (chan + channel_offset);
        }
    }

    Array2D<UVW<float>> Data::get_uvw(
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int baseline_offset,
        unsigned int time_offset,
        float integration_time) const
    {
        Array2D<UVW<float>> uvw(nr_baselines, nr_timesteps);
        get_uvw(uvw, baseline_offset, time_offset, integration_time);
        return uvw;
    }

    void Data::get_uvw(
        Array2D<UVW<float>>& uvw,
        unsigned int baseline_offset,
        unsigned int time_offset,
        float integration_time) const
    {
        unsigned int nr_baselines_total = m_baselines.size();
        unsigned int nr_baselines = uvw.get_y_dim();
        unsigned int nr_timesteps = uvw.get_x_dim();

        if (baseline_offset + nr_baselines > nr_baselines_total) {
            std::cerr << "Out-of-range baselines selected: ";
            if (baseline_offset > 0) {
                std::cerr << baseline_offset << " + " << nr_baselines;
            }
            std::cerr << nr_baselines
                      << " > " << nr_baselines_total << std::endl;
            nr_baselines = nr_baselines_total;
        }

        // Evaluate uvw per baseline
        #pragma omp parallel for
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            std::pair<float, Baseline> e = m_baselines[baseline_offset + bl];
            Baseline& baseline = e.second;

            for (unsigned time = 0; time < nr_timesteps; time++) {
                double u, v, w;
                evaluate_uvw(baseline, time_offset + time, integration_time, &u, &v, &w);
                uvw(bl, time) = {(float) u, (float) v, (float) w};
            } // end for time
        } // end for bl
    }

    float Data::get_max_uv() const
    {
        float max_uv = 0;
        for (auto baseline : m_baselines) {
            max_uv = std::max(max_uv, baseline.first);
        }
        return max_uv;
    }

    void Data::evaluate_uvw(
        Baseline& baseline,
        unsigned int time,
        float integration_time,
        double* u, double* v, double* w) const
    {
        unsigned station1 = baseline.station1;
        unsigned station2 = baseline.station2;
        double x1 = m_station_coordinates[station1].x;
        double y1 = m_station_coordinates[station1].y;
        double z1 = m_station_coordinates[station1].z;
        double x2 = m_station_coordinates[station2].x;
        double y2 = m_station_coordinates[station2].y;
        double z2 = m_station_coordinates[station2].z;
        double x[2] = {x1, x2};
        double y[2] = {y1, y2};
        double z[2] = {z1, z2};
        double seconds  = observation_seconds + (time * integration_time);
        double time_mjd = uvwsim_datetime_to_mjd(
                observation_year, observation_month, observation_day,
                observation_hour, observation_minute, seconds);
        uvwsim_evaluate_baseline_uvw(
                u, v, w,
                2, x, y, z,
                observation_ra, observation_dec, time_mjd);
    }

} // namespace idg


#include "datac.h"
