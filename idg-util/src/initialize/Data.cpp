#include <algorithm>

#include "uvwsim.h"

#include "Data.h"

namespace idg{

    Data::Data(
        unsigned int nr_stations_limit,
        unsigned int baseline_length_limit,
        std::string layout_file)
    {
        // Set station_coordinates
        set_station_coordinates(layout_file, nr_stations_limit);

        // Set baselines and max_uv
        set_baselines(station_coordinates, baseline_length_limit);
    }

    void Data::set_station_coordinates(
        std::string layout_file = "SKA1_low_ecef",
        unsigned int nr_stations_limit)
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

        // Limit number of stations
        if (nr_stations_limit > 0 && nr_stations_limit < nr_stations) {
            nr_stations = nr_stations_limit;
        }

        // Create vector of station coordinates
        for (unsigned i = 0; i < nr_stations; i++) {
            Data::StationCoordinate coordinate = { x[i], y[i], z[i] };
            station_coordinates.push_back(coordinate);
        }

        // Free memory
        free(x); free(y); free(z);
    }

    void Data::shuffle_stations() {
        auto nr_stations = station_coordinates.size();

        // Pick random station indices
        std::srand(0);
        std::vector<unsigned> station_indices(nr_stations);
        for (unsigned i = 0; i < nr_stations; i++) {
            station_indices[i] = i;
        }
        std::random_shuffle(station_indices.begin(), station_indices.end());

        std::vector<StationCoordinate> station_coordinates_shuffled;

        for (auto i : station_indices) {
            station_coordinates_shuffled.push_back(station_coordinates[i]);
        }

        std::swap(station_coordinates, station_coordinates_shuffled);
    }

    float Data::compute_image_size(unsigned grid_size)
    {
        return grid_size / max_uv * (SPEED_OF_LIGHT / start_frequency);
    }

    float Data::compute_grid_size(float image_size)
    {
        return (image_size * pixel_padding) / (SPEED_OF_LIGHT / start_frequency) * max_uv;
    };

    void Data::set_baselines(
        std::vector<StationCoordinate>& station_coordinates,
        unsigned int baseline_length_limit)
    {
        // Compute (maximum) baseline length and select baselines
        unsigned nr_stations = station_coordinates.size();
        max_uv = 0;
        for (unsigned station1 = 0; station1 < nr_stations; station1++) {
            for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
                Baseline baseline = {station1, station2};
                double u, v, w;

                // Compute uvw values for 12 hours of observation (with steps of 1 hours)
                // The baseline is only added when all samples are in range
                bool add_baseline = true;
                float max_uv_ = max_uv;
                for (unsigned time = 0; time < 12; time++) {
                    evaluate_uvw(baseline, time, 3600, &u, &v, &w);
                    float baseline_length = sqrtf(u*u + v*v);
                    max_uv_ = std::max(baseline_length, max_uv_);

                    if (baseline_length_limit > 0 && baseline_length > baseline_length_limit) {
                        add_baseline = false;
                        break;
                    }
                } // end for time

                // Add baseline
                if (add_baseline) {
                    baselines.push_back(std::pair<float, Baseline>(max_uv_, baseline));
                    max_uv = std::max(max_uv, max_uv_);
                }
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

        // Select the baselnes that fit in the grid
        bool selected_baselines[baselines.size()];

        // Keep track of stations corresponding to these baselines
        bool selected_stations[station_coordinates.size()];
        memset(selected_baselines, 0, sizeof(selected_baselines));

        #pragma omp parallel for
        for (unsigned i = 0; i < baselines.size(); i++) {
            std::pair<float, Baseline> entry = baselines[i];
            float baseline_length_meters = entry.first;
            float baseline_length_pixels = baseline_length_meters * image_size * (start_frequency / SPEED_OF_LIGHT);
            selected_baselines[i] = baseline_length_pixels < grid_size;
        }

        // Put all selected baselines into a new vector
        std::vector<std::pair<float, Baseline>> baselines_;

        float max_uv = 0;
        for (unsigned i = 0; i < baselines.size(); i++) {
            if (selected_baselines[i]) {
                baselines_.push_back(baselines[i]);
                float length =baselines[i].first;
                Baseline bl = baselines[i].second;
                max_uv = std::max(max_uv, length);
                selected_stations[bl.station1] = true;
                selected_stations[bl.station2] = true;
            }
        }

        // Put all selected stations into a new vector
        std::vector<StationCoordinate> station_coordinates_;

        for (unsigned i = 0; i < station_coordinates.size(); i++) {
            if (selected_stations[i]) {
                station_coordinates_.push_back(station_coordinates[i]);
            }
        }

        // Report
        //#if defined(DEBUG)
        std::cout << "number of stations: "
                  << station_coordinates.size() << " -> "
                  << station_coordinates_.size() << std::endl;
        std::cout << "number of baselines: "
                  << baselines.size() <<  " -> "
                  << baselines_.size() << std::endl;
        std::cout << "longest baseline = " << max_uv << std::endl;
        //#endif

        // Replace members
        baselines = baselines_;
        station_coordinates = station_coordinates_;

    }

    void Data::get_frequencies(
        Array1D<float>& frequencies,
        float image_size,
        unsigned int channel_offset) const
    {
        auto nr_channels = frequencies.get_x_dim();
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
        unsigned int nr_baselines_total = baselines.size();
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
            std::pair<float, Baseline> e = baselines[baseline_offset + bl];
            Baseline& baseline = e.second;

            for (unsigned time = 0; time < nr_timesteps; time++) {
                double u, v, w;
                evaluate_uvw(baseline, time_offset + time, integration_time, &u, &v, &w);
                uvw(bl, time) = {(float) u, (float) v, (float) w};
            } // end for time
        } // end for bl
    }

    void Data::evaluate_uvw(
        Baseline& baseline,
        unsigned int time,
        float integration_time,
        double* u, double* v, double* w) const
    {
        unsigned station1 = baseline.station1;
        unsigned station2 = baseline.station2;
        double x1 = station_coordinates[station1].x;
        double y1 = station_coordinates[station1].y;
        double z1 = station_coordinates[station1].z;
        double x2 = station_coordinates[station2].x;
        double y2 = station_coordinates[station2].y;
        double z2 = station_coordinates[station2].z;
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
