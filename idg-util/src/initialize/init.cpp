#include "init.h"

#include "uvwsim.h"

namespace idg {

    void init_example_uvw(
        void *ptr,
        int nr_stations,
        int nr_baselines,
        int nr_time,
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
        int nr_stations_file = uvwsim_get_num_stations(filename);

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

        if (uvwsim_load_station_coords(filename, nr_stations_file, x, y, z) != nr_stations_file) {
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
            int i = 0;
            srandom(RANDOM_SEED);
            while (i < nr_stations) {
                int index = nr_stations_file * ((double) random() / RAND_MAX);
                bool found = true;
                for (int j = 0; j < i; j++) {
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
            for (int i = 0; i < nr_stations; i++) {
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
        double obs_length_hours = (nr_time * integration_time) / (3600.0);
        double obs_length_days = obs_length_hours / 24.0;

        // Allocate memory for baseline coordinates
        int nr_coordinates = nr_time * nr_baselines;
        double *uu = (double*) malloc(nr_coordinates * sizeof(double));
        double *vv = (double*) malloc(nr_coordinates * sizeof(double));
        double *ww = (double*) malloc(nr_coordinates * sizeof(double));

        // Evaluate baseline uvw coordinates.
        #pragma omp parallel for
        for (int t = 0; t < nr_time; t++) {
            double time_mjd = start_time_mjd + t
                              * (obs_length_days/(double)nr_time);
            size_t offset = t * nr_baselines;
            uvwsim_evaluate_baseline_uvw(
                &uu[offset], &vv[offset], &ww[offset],
                nr_stations, x, y, z, ra0, dec0, time_mjd);
        }

        // Fill UVW datastructure
        #pragma omp parallel for
        for (int bl = 0; bl < nr_baselines; bl++) {
            for (int t = 0; t < nr_time; t++) {
                int src = t * nr_baselines + bl;
                int dst = bl * nr_time + t;
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
        int   size_in,
        float *__restrict__ spheroidal_out,
        int   size_out)
    {
        auto in_ft  = new std::complex<float>[size_in*size_in];
        auto out_ft = new std::complex<float>[size_out*size_out];

        for (int i = 0; i < size_in; i++) {
            for (int j = 0; j < size_in; j++) {
                in_ft[i*size_in + j] = spheroidal_in[i*size_in + j];
            }
        }
        idg::fft2f(size_in, in_ft);

        int offset = int((size_out - size_in)/2);

        for (int i = 0; i < size_in; i++) {
            for (int j = 0; j < size_in; j++) {
                out_ft[(i+offset)*size_out + (j+offset)] = in_ft[i*size_in + j];
            }
        }
        idg::ifft2f(size_out, out_ft);

        float s = 1.0f / (size_in * size_in);
        for (int i = 0; i < size_out; i++) {
            for (int j = 0; j < size_out; j++) {
                spheroidal_out[i*size_out + j] = out_ft[i*size_out + j].real() * s;
            }
        }

        delete [] in_ft;
        delete [] out_ft;
    }

    Array1D<float> get_example_frequencies(
        unsigned int nr_channels,
        float start_frequency,
        float frequency_increment)
    {
        Array1D<float> frequencies(nr_channels);

        for (int chan = 0; chan < nr_channels; chan++) {
            frequencies(chan) = start_frequency + frequency_increment * chan;
        }

        return frequencies;
    }

    Array1D<float> get_example_wavenumbers(
        unsigned int nr_channels,
        float start_frequency,
        float frequency_increment)
    {
        Array1D<float> frequencies = get_example_frequencies(nr_channels);
        Array1D<float> wavenumbers(nr_channels);

        const double speed_of_light = 299792458.0;
        for (int i = 0; i < nr_channels; i++) {
            wavenumbers(i) =  2 * M_PI * frequencies(i) / speed_of_light;
        }

        return wavenumbers;
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
        for (int bl = 0; bl < nr_baselines; bl++) {
            for (int time = 0; time < nr_timesteps; time++) {
                for (int chan = 0; chan < nr_channels; chan++) {
                    visibilities(bl, time, chan) = visibility;
                }
            }
        }

        return visibilities;
    }


    Array1D<std::pair<unsigned int,unsigned int>> get_example_baselines(
        unsigned int nr_stations,
        unsigned int nr_baselines)
    {
        Array1D<std::pair<unsigned int,unsigned int>> baselines(nr_baselines);

        int bl = 0;

        for (int station1 = 0 ; station1 < nr_stations; station1++) {
            for (int station2 = station1 + 1; station2 < nr_stations; station2++) {
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


    Array4D<Matrix2x2<std::complex<float>>> get_example_aterms(
        unsigned int nr_timeslots,
        unsigned int nr_stations,
        unsigned int height,
        unsigned int width)
    {
        Array4D<Matrix2x2<std::complex<float>>> aterms(nr_timeslots, nr_stations, height, width);
        const Matrix2x2<std::complex<float>> aterm = {1.0f, 0.0f, 0.0f, 1.0f};

        for (int t = 0; t < nr_timeslots; t++) {
            for (int ant = 0; ant < nr_stations; ant++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
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

        for (int time = 0; time < nr_timeslots; time++) {
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

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
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
        for (int i = 0; i < height; i++) {
            float tmp = fabs(-1 + i*2.0f/float(height));
            y[i] = evaluate_spheroidal(tmp);
        }

        // Evaluate columns
        float x[width];
        for (int i = 0; i < width; i++) {
            float tmp = fabs(-1 + i*2.0f/float(width));
            x[i] = evaluate_spheroidal(tmp);
        }

        // Set values
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                 spheroidal(i, j) = y[i]*x[j];
            }
        }

        return spheroidal;
    }

} // namespace idg


#include "initc.h"
