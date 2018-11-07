#include <ctime> // time
#include <algorithm> // random_shuffle

#include "init.h"

#include "uvwsim.h"

namespace idg {

    void init_example_uvw(
        void *ptr,
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time)
    {
        UVWCoordinate<float> *uvw = (UVWCoordinate<float> *) ptr;

        // Try to load layout file from environment
        char *cstr_layout_file = getenv(ENV_LAYOUT_FILE);

        // Check whether layout file exists
        char filename[512];
        if (cstr_layout_file) {
            sprintf(filename, "%s/%s", IDG_DATA_DIR, cstr_layout_file);
        } else {
            sprintf(filename, "%s/%s", IDG_DATA_DIR, LAYOUT_FILE);
        }

        if (!uvwsim_file_exists(filename)) {
            std::cerr << "Unable to find specified layout file: "
                      << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        // Read the number of stations in the layout file.
        unsigned nr_stations_file = uvwsim_get_num_stations(filename);

        // Check wheter the requested number of station is feasible
        if (nr_stations_file < nr_stations) {
           std::cerr << "More stations requested than present in layout file: "
                     << "(" << nr_stations_file << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Allocate memory for antenna coordinates
        double *x = (double*) malloc(nr_stations_file * sizeof(double));
        double *y = (double*) malloc(nr_stations_file * sizeof(double));
        double *z = (double*) malloc(nr_stations_file * sizeof(double));

        // Load the antenna coordinates
        #if defined(DEBUG)
        printf("looking for stations file in: %s\n", filename);
        #endif

        unsigned nr_stations_read = uvwsim_load_station_coords(filename, nr_stations_file, x, y, z);
        if (nr_stations_read != nr_stations_file) {
            std::cerr << "Failed to read antenna coordinates." << std::endl;
            exit(EXIT_FAILURE);
        }

        // Select some antennas randomly when not all antennas are requested
        if (nr_stations < nr_stations_file) {
            // Allocate memory for selection of antenna coordinates
            double *_x = (double*) malloc(nr_stations * sizeof(double));
            double *_y = (double*) malloc(nr_stations * sizeof(double));
            double *_z = (double*) malloc(nr_stations * sizeof(double));

            // Generate nr_stations random numbers
            int station_number[nr_stations];
            unsigned i = 0;
            srandom(RANDOM_SEED);
            while (i < nr_stations) {
                int index = nr_stations_file * ((double) random() / RAND_MAX);
                bool found = true;
                for (unsigned j = 0; j < i; j++) {
                    if (station_number[j] == index) {
                        found = false;
                        break;
                    }
                }
                if (found) {
                     station_number[i++] = index;
                }
             }

            // Set stations
            for (unsigned i = 0; i < nr_stations; i++) {
                _x[i] = x[station_number[i]];
                _y[i] = y[station_number[i]];
                _z[i] = z[station_number[i]];
            }

            // Swap pointers and free memory
            double *__x = x;
            double *__y = y;
            double *__z = z;
            x = _x;
            y = _y;
            z = _z;
            free(__x);
            free(__y);
            free(__z);
        }

        // Define observation parameters
        double ra0  = RIGHT_ASCENSION;
        double dec0 = DECLINATION;
        double start_time_mjd = uvwsim_datetime_to_mjd(YEAR, MONTH, DAY, HOUR, MINUTE, SECONDS);
        double obs_length_hours = (nr_timesteps * integration_time) / (3600.0);
        double obs_length_days = obs_length_hours / 24.0;

        // Allocate memory for baseline coordinates
        int nr_coordinates = nr_timesteps * nr_baselines;
        double *uu = (double*) malloc(nr_coordinates * sizeof(double));
        double *vv = (double*) malloc(nr_coordinates * sizeof(double));
        double *ww = (double*) malloc(nr_coordinates * sizeof(double));

        // Evaluate baseline uvw coordinates.
        #pragma omp parallel for
        for (unsigned t = 0; t < nr_timesteps; t++) {
            double time_mjd = start_time_mjd + t
                              * (obs_length_days/(double)nr_timesteps);
            size_t offset = t * nr_baselines;
            uvwsim_evaluate_baseline_uvw(
                &uu[offset], &vv[offset], &ww[offset],
                nr_stations, x, y, z, ra0, dec0, time_mjd);
        }

        // Fill UVW datastructure
        #pragma omp parallel for
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            for (unsigned t = 0; t < nr_timesteps; t++) {
                int src = t * nr_baselines + bl;
                int dst = bl * nr_timesteps + t;
                UVWCoordinate<float> value = {
                    (float) uu[src], (float) vv[src], (float) ww[src]};
                uvw[dst] = value;
            }
        }

        // Free memory
        free(x); free(y); free(z);
        free(uu); free(vv); free(ww);
    }


    // Function to compute spheroidal. Based on reference code by BvdT.
    float evaluate_spheroidal(float nu)
    {
        float P[2][5] = {
            {8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
            {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
        float Q[2][3] = {
            {1.0000000e0, 8.212018e-1, 2.078043e-1},
            {1.0000000e0, 9.599102e-1, 2.918724e-1}};

        int part;
        float end;
        if (nu >= 0.0 && nu < 0.75) {
            part = 0;
            end  = 0.75f;
        } else if (nu >= 0.75 && nu <= 1.00) {
            part = 1;
            end  = 1.0f;
        } else {
            return 0.0f;
        }

        float nusq = nu * nu;
        float delnusq = nusq - end * end;
        float delnusqPow = delnusq;
        float top = P[part][0];
        for (auto k = 1; k < 5; k++) {
            top += P[part][k] * delnusqPow;
            delnusqPow *= delnusq;
        }

        float bot = Q[part][0];
        delnusqPow = delnusq;
        for (auto k = 1; k < 3; k++) {
            bot += Q[part][k] * delnusqPow;
            delnusqPow *= delnusq;
        }

        if (bot == 0.0f) {
            return 0.0f;
        } else {
            return (1.0 - nusq) * (top / bot);
        }
    }

    // TODO: make generic, not spheroidal specific
    // TODO: use real-to-complex and complex-to-real FFT
    void resize_spheroidal(
        float *__restrict__ spheroidal_in,
        unsigned int        size_in,
        float *__restrict__ spheroidal_out,
        unsigned int        size_out)
    {
        auto in_ft  = new std::complex<float>[size_in*size_in];
        auto out_ft = new std::complex<float>[size_out*size_out];

        for (unsigned i = 0; i < size_in; i++) {
            for (unsigned j = 0; j < size_in; j++) {
                in_ft[i*size_in + j] = spheroidal_in[i*size_in + j];
            }
        }
        idg::fft2f(size_in, in_ft);

        int offset = int((size_out - size_in)/2);

        for (unsigned i = 0; i < size_in; i++) {
            for (unsigned j = 0; j < size_in; j++) {
                out_ft[(i+offset)*size_out + (j+offset)] = in_ft[i*size_in + j];
            }
        }
        idg::ifft2f(size_out, out_ft);

        float s = 1.0f / (size_in * size_in);
        for (unsigned i = 0; i < size_out; i++) {
            for (unsigned j = 0; j < size_out; j++) {
                spheroidal_out[i*size_out + j] = out_ft[i*size_out + j].real() * s;
            }
        }

        delete [] in_ft;
        delete [] out_ft;
    }


    /*
     * Memory-allocation is handled by Proxy
     */
    Array1D<float> get_example_frequencies(
        proxy::Proxy& proxy,
        unsigned int nr_channels,
        float start_frequency,
        float frequency_increment)
    {
        auto bytes = auxiliary::sizeof_wavenumbers(nr_channels);
        float* ptr = (float *) proxy.allocate_memory(bytes);

        Array1D<float> frequencies(ptr, nr_channels);

        for (unsigned chan = 0; chan < nr_channels; chan++) {
            frequencies(chan) = start_frequency + frequency_increment * chan;
        }

        return frequencies;
    }

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        proxy::Proxy& proxy,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_channels)
    {
        auto bytes = auxiliary::sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
        Visibility<std::complex<float>>* ptr = (Visibility<std::complex<float>>*) proxy.allocate_memory(bytes);

        Array3D<Visibility<std::complex<float>>> visibilities(ptr, nr_baselines, nr_timesteps, nr_channels);
        const Visibility<std::complex<float>> visibility = {1.0f, 0.0f, 0.0f, 1.0f};

        // Set all visibilities
        #pragma omp parallel for
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            for (unsigned time = 0; time < nr_timesteps; time++) {
                for (unsigned chan = 0; chan < nr_channels; chan++) {
                    visibilities(bl, time, chan) = visibility;
                }
            }
        }

        return visibilities;
    }

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        proxy::Proxy& proxy,
        unsigned int nr_stations,
        unsigned int nr_baselines)
    {
        auto bytes = auxiliary::sizeof_baselines(nr_baselines);
        std::pair<unsigned int, unsigned int>* ptr = (std::pair<unsigned int, unsigned int>*) proxy.allocate_memory(bytes);
        Array1D<std::pair<unsigned int,unsigned int>> baselines(ptr, nr_baselines);

        unsigned bl = 0;

        for (unsigned station1 = 0 ; station1 < nr_stations; station1++) {
            for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
                if (bl >= nr_baselines) {
                    break;
                }
                baselines(bl) = std::pair<unsigned int,unsigned int>(station1, station2);
                bl++;
            }
        }

        return baselines;
    }

    Array2D<UVWCoordinate<float>> get_example_uvw(
        proxy::Proxy& proxy,
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time)
    {
        auto bytes = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
        UVWCoordinate<float>* ptr = (UVWCoordinate<float>*) proxy.allocate_memory(bytes);

        Array2D<UVWCoordinate<float>> uvw(ptr, nr_baselines, nr_timesteps);
        void *uvw_ptr = uvw.data();
        init_example_uvw(uvw_ptr, nr_stations, nr_baselines, nr_timesteps, integration_time);

        return uvw;
    }

    Array3D<std::complex<float>> get_zero_grid(
        proxy::Proxy& proxy,
        unsigned int nr_correlations,
        unsigned int height,
        unsigned int width)
    {
        assert(height == width);
        auto bytes = auxiliary::sizeof_grid(height);
        std::complex<float>* ptr = (std::complex<float>*) proxy.allocate_memory(bytes);

        Array3D<std::complex<float>> grid(ptr, nr_correlations, height, width);
        memset(grid.data(), 0, grid.bytes());
        return grid;
    }

    Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width)
    {
        assert(height == width);
        auto bytes = auxiliary::sizeof_aterms(nr_stations, nr_timeslots, height);
        Matrix2x2<std::complex<float>>* ptr = (Matrix2x2<std::complex<float>>*) proxy.allocate_memory(bytes);

        Array4D<Matrix2x2<std::complex<float>>> aterms(ptr, nr_timeslots, nr_stations, height, width);
        const Matrix2x2<std::complex<float>> aterm = {1.0f, 0.0f, 0.0f, 1.0f};

        for (unsigned t = 0; t < nr_timeslots; t++) {
            for (unsigned ant = 0; ant < nr_stations; ant++) {
                for (unsigned y = 0; y < height; y++) {
                    for (unsigned x = 0; x < width; x++) {
                        aterms(t, ant, y, x) = aterm;
                    }
                }
            }
        }

        return aterms;
    }

    Array1D<unsigned int> get_example_aterms_offsets(
        proxy::Proxy& proxy,
        unsigned int nr_timeslots,
        unsigned int nr_timesteps)
    {
        auto bytes = auxiliary::sizeof_aterms_offsets(nr_timeslots);
        unsigned int* ptr = (unsigned int*) proxy.allocate_memory(bytes);

        Array1D<unsigned int> aterms_offsets(ptr, nr_timeslots + 1);

        for (unsigned time = 0; time < nr_timeslots; time++) {
             aterms_offsets(time) = time * (nr_timesteps / nr_timeslots);
        }

        aterms_offsets(nr_timeslots) = nr_timesteps;

        return aterms_offsets;
    }

    Array2D<float> get_example_spheroidal(
        proxy::Proxy& proxy,
        unsigned int height,
        unsigned int width)
    {
        assert(height == width);
        auto bytes = auxiliary::sizeof_spheroidal(height);
        float* ptr = (float*) proxy.allocate_memory(bytes);

        Array2D<float> spheroidal(ptr, height, width);

        // Evaluate rows
        float y[height];
        for (unsigned i = 0; i < height; i++) {
            float tmp = fabs(-1 + i*2.0f/float(height));
            y[i] = evaluate_spheroidal(tmp);
        }

        // Evaluate columns
        float x[width];
        for (unsigned i = 0; i < width; i++) {
            float tmp = fabs(-1 + i*2.0f/float(width));
            x[i] = evaluate_spheroidal(tmp);
        }

        // Set values
        for (unsigned i = 0; i < height; i++) {
            for (unsigned j = 0; j < width; j++) {
                 spheroidal(i, j) = y[i]*x[j];
            }
        }

        return spheroidal;
    }


    /*
     * Default memory allocation
     */
    Array1D<float> get_example_frequencies(
        unsigned int nr_channels,
        float start_frequency,
        float frequency_increment)
    {
        Array1D<float> frequencies(nr_channels);

        for (unsigned chan = 0; chan < nr_channels; chan++) {
            frequencies(chan) = start_frequency + frequency_increment * chan;
        }

        return frequencies;
    }

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        unsigned int nr_channels)
    {
        Array3D<Visibility<std::complex<float>>> visibilities(nr_baselines, nr_timesteps, nr_channels);
        const Visibility<std::complex<float>> visibility = {1.0f, 0.0f, 0.0f, 1.0f};

        // Set all visibilities
        #pragma omp parallel for
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            for (unsigned time = 0; time < nr_timesteps; time++) {
                for (unsigned chan = 0; chan < nr_channels; chan++) {
                    visibilities(bl, time, chan) = visibility;
                }
            }
        }

        return visibilities;
    }

    Array3D<Visibility<std::complex<float>>> get_example_visibilities(
        Array2D<UVWCoordinate<float>> &uvw,
        Array1D<float> &frequencies,
        float        image_size,
        unsigned int grid_size,
        unsigned int nr_point_sources,
        unsigned int max_pixel_offset,
        unsigned int random_seed,
        float        amplitude)
    {
        unsigned int nr_baselines = uvw.get_y_dim();
        unsigned int nr_timesteps = uvw.get_x_dim();
        unsigned int nr_channels  = frequencies.get_x_dim();

        Array3D<Visibility<std::complex<float>>> visibilities(nr_baselines, nr_timesteps, nr_channels);

        srand(random_seed);

        for (unsigned i = 0; i < nr_point_sources; i++) {
            float x_offset = (random() * (max_pixel_offset)) - (max_pixel_offset/2);
            float y_offset = (random() * (max_pixel_offset)) - (max_pixel_offset/2);

            add_pt_src(visibilities, uvw, frequencies, image_size, grid_size, x_offset, y_offset, amplitude);
        }

        return visibilities;
    }

    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        unsigned int nr_stations,
        unsigned int nr_baselines)
    {
        Array1D<std::pair<unsigned int,unsigned int>> baselines(nr_baselines);

        unsigned bl = 0;

        for (unsigned station1 = 0 ; station1 < nr_stations; station1++) {
            for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
                if (bl >= nr_baselines) {
                    break;
                }
                baselines(bl) = std::pair<unsigned int,unsigned int>(station1, station2);
                bl++;
            }
        }

        return baselines;
    }


    Array2D<UVWCoordinate<float>> get_example_uvw(
        unsigned int nr_stations,
        unsigned int nr_baselines,
        unsigned int nr_timesteps,
        float integration_time)
    {
        Array2D<UVWCoordinate<float>> uvw(nr_baselines, nr_timesteps);
        void *uvw_ptr = uvw.data();
        init_example_uvw(uvw_ptr, nr_stations, nr_baselines, nr_timesteps, integration_time);

        return uvw;
    }


    Array3D<std::complex<float>> get_zero_grid(
        unsigned int nr_correlations,
        unsigned int height,
        unsigned int width
    ) {
        Array3D<std::complex<float>> grid(nr_correlations, height, width);
        memset(grid.data(), 0, grid.bytes());
        return grid;
    }


    Array4D<Matrix2x2<std::complex<float>>> get_identity_aterms(
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width)
    {
        Array4D<Matrix2x2<std::complex<float>>> aterms(nr_timeslots, nr_stations, height, width);
        const Matrix2x2<std::complex<float>> aterm = {1.0f, 0.0f, 0.0f, 1.0f};

        for (unsigned t = 0; t < nr_timeslots; t++) {
            for (unsigned ant = 0; ant < nr_stations; ant++) {
                for (unsigned y = 0; y < height; y++) {
                    for (unsigned x = 0; x < width; x++) {
                        aterms(t, ant, y, x) = aterm;
                    }
                }
            }
        }

        return aterms;
    }


    Array1D<unsigned int> get_example_aterms_offsets(
        unsigned int nr_timeslots,
        unsigned int nr_timesteps)
    {
        Array1D<unsigned int> aterms_offsets(nr_timeslots + 1);

        for (unsigned time = 0; time < nr_timeslots; time++) {
             aterms_offsets(time) = time * (nr_timesteps / nr_timeslots);
        }

        aterms_offsets(nr_timeslots) = nr_timesteps;

        return aterms_offsets;
    }

    Array2D<float> get_identity_spheroidal(
        unsigned int height,
        unsigned int width)
    {
        Array2D<float> spheroidal(height, width);

        float value = 1.0;

        for (unsigned y = 0; y < height; y++) {
            for (unsigned x = 0; x < width; x++) {
                 spheroidal(y, x) = value;
            }
        }

        return spheroidal;
    }

    Array2D<float> get_example_spheroidal(
        unsigned int height,
        unsigned int width)
    {
        Array2D<float> spheroidal(height, width);

        // Evaluate rows
        float y[height];
        for (unsigned i = 0; i < height; i++) {
            float tmp = fabs(-1 + i*2.0f/float(height));
            y[i] = evaluate_spheroidal(tmp);
        }

        // Evaluate columns
        float x[width];
        for (unsigned i = 0; i < width; i++) {
            float tmp = fabs(-1 + i*2.0f/float(width));
            x[i] = evaluate_spheroidal(tmp);
        }

        // Set values
        for (unsigned i = 0; i < height; i++) {
            for (unsigned j = 0; j < width; j++) {
                 spheroidal(i, j) = y[i]*x[j];
            }
        }

        return spheroidal;
    }

    void add_pt_src(
        Array3D<Visibility<std::complex<float>>> &visibilities,
        Array2D<UVWCoordinate<float>> &uvw,
        Array1D<float> &frequencies,
        float          image_size,
        unsigned int   grid_size,
        float          x,
        float          y,
        float          amplitude)
    {
        auto nr_baselines = visibilities.get_z_dim();
        auto nr_timesteps = visibilities.get_y_dim();
        auto nr_channels  = visibilities.get_x_dim();

        float l = x * image_size/grid_size;
        float m = y * image_size/grid_size;

        #pragma omp parallel for
        for (unsigned b = 0; b < nr_baselines; b++) {
            for (unsigned t = 0; t < nr_timesteps; t++) {
                for (unsigned c = 0; c < nr_channels; c++) {
                    float u = frequencies(c) * uvw(b,t).u / (SPEED_OF_LIGHT);
                    float v = frequencies(c) * uvw(b,t).v / (SPEED_OF_LIGHT);
                    std::complex<float> value = amplitude *
                        std::exp(std::complex<float>(0, -2 * M_PI * (u*l + v*m)));
                    visibilities(b,t,c).xx += value;
                    visibilities(b,t,c).xy += value;
                    visibilities(b,t,c).yx += value;
                    visibilities(b,t,c).yy += value;
                }
            }
        }
    }

    /*
     * Class Data
     */
    Data::Data(
        unsigned int grid_size,
        unsigned int nr_stations_limit,
        unsigned int baseline_length_limit,
        std::string layout_file,
        float start_frequency
    ) : start_frequency(start_frequency)
    {

        // Set station_coordinates
        set_station_coordinates(layout_file, nr_stations_limit);

        // Set baselines anx max_baseline_length
        set_baselines(station_coordinates, baseline_length_limit);

        // Set derived parameters
        image_size = (grid_size * pixel_padding) / max_baseline_length * (start_frequency / SPEED_OF_LIGHT);
        frequency_increment = SPEED_OF_LIGHT / (max_baseline_length * image_size);
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

        // Pick random station indices
        std::srand(0);
        std::vector<unsigned> station_indices(nr_stations);
        for (unsigned i = 0; i < nr_stations; i++) {
            station_indices[i] = i;
        }
        std::random_shuffle(station_indices.begin(), station_indices.end());

        // Limit number of stations
        if (nr_stations_limit > 0 && nr_stations_limit < nr_stations) {
            nr_stations = nr_stations_limit;
        }

        // Create vector of shuffled station indices
        for (unsigned i = 0; i < nr_stations; i++) {
            unsigned station_index = station_indices[i];
            Data::StationCoordinate coordinate = { x[station_index], y[station_index], z[station_index] };
            station_coordinates.push_back(coordinate);
        }

        // Free memory
        free(x); free(y); free(z);
    }


    void Data::set_baselines(
        std::vector<StationCoordinate>& station_coordinates,
        unsigned int baseline_length_limit)
    {
        // Compute (maximum) baseline length and select baselines
        unsigned nr_stations = station_coordinates.size();
        max_baseline_length = 0;
        for (unsigned station1 = 0; station1 < nr_stations; station1++) {
            for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
                double x1 = station_coordinates[station1].x;
                double y1 = station_coordinates[station1].y;
                double z1 = station_coordinates[station1].z;
                double x2 = station_coordinates[station2].x;
                double y2 = station_coordinates[station2].y;
                double z2 = station_coordinates[station2].z;
                double x_distance = abs(x1 - x2);
                double y_distance = abs(y1 - y2);
                double z_distance = abs(z1 - z2);
                double baseline_length = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
                if (baseline_length < baseline_length_limit) {
                    baselines.push_back({station1, station2});
                    if (baseline_length > max_baseline_length) {
                        max_baseline_length = baseline_length;
                    }
                }
            }
        }
    }


    void Data::get_frequencies(
        Array1D<float>& frequencies,
        unsigned int channel_offset) const
    {
        auto nr_channels = frequencies.get_x_dim();

        for (unsigned chan = 0; chan < nr_channels; chan++) {
            frequencies(chan) = start_frequency + frequency_increment * (chan + channel_offset);
        }
    }

    void Data::get_uvw(
        Array2D<UVWCoordinate<float>>& uvw,
        unsigned int baseline_offset,
        unsigned int time_offset,
        float integration_time) const
    {
        unsigned int nr_baselines_total = baselines.size();
        unsigned int nr_baselines = uvw.get_y_dim();
        unsigned int nr_timesteps = uvw.get_x_dim();

        if (baseline_offset + nr_baselines > nr_baselines_total) {
            std::cerr << "Out-of-range baselines selected: "
                      << baseline_offset + " + " + nr_baselines
                      << " > " << nr_baselines_total << std::endl;
        }

        // Compute observation start time
        unsigned int year     = observation_year;
        unsigned int month    = observation_month;
        unsigned int day      = observation_day;
        unsigned int hour     = observation_hour;
        unsigned int minute   = observation_minute;

        // Define observation parameters
        double ra0  = observation_ra;
        double dec0 = observation_dec;

        // Evaluate uvw per baseline
        #pragma omp parallel for
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            Baseline baseline = baselines[baseline_offset + bl];
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
            double u, v, w;

            for (unsigned time = 0; time < nr_timesteps; time++) {
                double seconds  = observation_seconds + ((time_offset + time) * integration_time);
                double time_mjd = uvwsim_datetime_to_mjd(year, month, day, hour, minute, seconds);

                uvwsim_evaluate_baseline_uvw(
                        &u, &v, &w,
                        2, x, y, z, ra0, dec0, time_mjd);
                uvw(bl, time) = {(float) u, (float) v, (float) w};
            } // end for time
        } // end for bl
    }

} // namespace idg


#include "initc.h"
