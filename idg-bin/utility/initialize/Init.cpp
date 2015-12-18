#include "Init.h"

#define TYPEDEF_UVW               typedef struct { float u, v, w; } UVW;
#define TYPEDEF_UVW_TYPE          typedef UVW UVWType[nr_baselines][nr_time];
#define TYPEDEF_VISIBILITIES_TYPE typedef std::complex<float> VisibilitiesType[nr_baselines][nr_time][nr_channels][nr_polarizations];
#define TYPEDEF_WAVENUMBER_TYPE   typedef float WavenumberType[nr_channels];
#define TYPEDEF_ATERM_TYPE        typedef std::complex<float> ATermType[nr_stations][nr_timeslots][nr_polarizations][subgridsize][subgridsize];
#define TYPEDEF_SPHEROIDAL_TYPE   typedef float SpheroidalType[subgridsize][subgridsize];
#define TYPEDEF_BASELINE          typedef struct { int station1, station2; } Baseline;
#define TYPEDEF_BASELINE_TYPE     typedef Baseline BaselineType[nr_baselines];
#define TYPEDEF_SUBGRID_TYPE      typedef std::complex<float> SubGridType[nr_baselines][nr_chunks][subgridsize][subgridsize][nr_polarizations];
#define TYPEDEF_GRID_TYPE         typedef std::complex<float> GridType[nr_polarizations][gridsize][gridsize];
#define TYPEDEF_COORDINATE        typedef struct { int x, y; } Coordinate;
#define TYPEDEF_METADATA          typedef struct { int time_nr; Baseline baseline; Coordinate coordinate; } Metadata;
#define TYPEDEF_METADATA_TYPE     typedef Metadata MetadataType[nr_subgrids];


namespace idg {

/* Methods where pointed to allocated memory is provided */
    void init_uvw(void *ptr, int nr_stations, int nr_baselines, int nr_time, int integration_time) {
        TYPEDEF_UVW
        TYPEDEF_UVW_TYPE

        UVWType *uvw = (UVWType *) ptr;

        // Check whether layout file exists
        bool found = false;
        char filename[512];
        sprintf(filename, "%s/%s/%s", IDG_SOURCE_DIR, LAYOUT_DIR, LAYOUT_FILE);

        if (!uvwsim_file_exists(filename)) {
            std::cerr << "Unable to find specified layout file: "
                      << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        // Read the number of stations in the layout file.
        int nr_stations_file = uvwsim_get_num_stations(filename);

        // Check wheter the requested number of station is feasible
        // if (nr_stations_file < nr_stations) {
        //    std::cerr << "More stations requested than present in layout file: "
        //              << "(" << nr_stations_file << ")" << std::endl;
        // }

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
        for (int t = 0; t < nr_time; t++) {
            double time_mjd = start_time_mjd + t
                              * (obs_length_days/(double)nr_time);
            size_t offset = t * nr_baselines;
            uvwsim_evaluate_baseline_uvw(
                &uu[offset], &vv[offset], &ww[offset],
                nr_stations, x, y, z, ra0, dec0, time_mjd);
        }

        // Fill UVW datastructure
        for (int bl = 0; bl < nr_baselines; bl++) {
            for (int t = 0; t < nr_time; t++) {
                int i = t * nr_baselines + bl;
                UVW value = {(float) uu[i], (float) vv[i], (float) ww[i]};
                (*uvw)[bl][t] = value;
            }
        }

        // Free memory
        free(x); free(y); free(z);
        free(uu); free(vv); free(ww);
    }



void init_visibilities(void *ptr, int nr_baselines, int nr_time,
                       int nr_channels, int nr_polarizations) {
    TYPEDEF_VISIBILITIES_TYPE
	VisibilitiesType *visibilities = (VisibilitiesType *) (ptr);

	// Fixed visibility
    std::complex<float> visibility(1, 0);

	// Set all visibilities
	for (int bl = 0; bl < nr_baselines; bl++) {
		for (int time = 0; time < nr_time; time++) {
			for (int chan = 0; chan < nr_channels; chan++) {
				for (int pol = 0; pol < nr_polarizations; pol++) {
					(*visibilities)[bl][time][chan][pol] = visibility;
				}
			}
		}
	}
}

void init_wavenumbers(void *ptr, int nr_channels) {
	TYPEDEF_WAVENUMBER_TYPE
    WavenumberType *wavenumbers = (WavenumberType *) ptr;

	// Initialize frequencies
	float frequencies[nr_channels];
	for (int chan = 0; chan < nr_channels; chan++) {
		frequencies[chan] = START_FREQUENCY + FREQUENCY_INCREMENT * chan;
	}

	// Initialize wavenumbers
	for (int i = 0; i < nr_channels; i++) {
		(*wavenumbers)[i] =  2 * M_PI * frequencies[i] / SPEED_OF_LIGHT;
	}
}

void init_aterm(void *ptr, int nr_stations, int nr_timeslots,
                int nr_polarizations, int subgridsize) {
	TYPEDEF_ATERM_TYPE
    ATermType *aterm = (ATermType *) ptr;

    std::complex<float> value(1, 1);

	for (int ant = 0; ant < nr_stations; ant++) {
        for (int t = 0; t < nr_timeslots; t++) {
		    for (int y = 0; y < subgridsize; y++) {
		    	for (int x = 0; x < subgridsize; x++) {
		    		for (int pol = 0; pol < nr_polarizations; pol++) {
		    			(*aterm)[ant][t][pol][y][x] = value;
		    		}
		    	}
		    }
        }
	}
}

void init_spheroidal(void *ptr, int subgridsize) {
    TYPEDEF_SPHEROIDAL_TYPE
    SpheroidalType *spheroidal = (SpheroidalType *) ptr;

	float value = 1.0;

	for (int y = 0; y < subgridsize; y++) {
		for (int x = 0; x < subgridsize; x++) {
			(*spheroidal)[y][x] = value;
		}
	}
}

void init_baselines(void *ptr, int nr_stations, int nr_baselines) {
    TYPEDEF_BASELINE
    TYPEDEF_BASELINE_TYPE
	BaselineType *baselines = (BaselineType *) ptr;

	int bl = 0;

	for (int station1 = 1 ; station1 < nr_stations; station1++) {
		for (int station2 = 0; station2 < station1; station2++) {
			if (bl >= nr_baselines) {
				break;
			}
			(*baselines)[bl].station1 = station1;
			(*baselines)[bl].station2 = station2;
			bl++;
		}
	}
}

void init_subgrid(void *ptr, int nr_baselines, int subgridsize, int nr_polarizations, int nr_chunks) {
	TYPEDEF_SUBGRID_TYPE
    SubGridType *subgrid = (SubGridType *) ptr;
	memset(subgrid, 0, sizeof(SubGridType));
}

void init_grid(void *ptr, int gridsize, int nr_polarizations) {
    TYPEDEF_GRID_TYPE
    GridType *grid = (GridType *) ptr;
	memset(grid, 0, sizeof(GridType));
}

float min(float coordinate_first, float coordinate_last, float wavenumber_first, float wavenumber_last) {
    return MIN(
           MIN(coordinate_first * wavenumber_first, coordinate_first * wavenumber_last),
           MIN(coordinate_last  * wavenumber_first, coordinate_last  * wavenumber_last));
}

float max(float coordinate_first, float coordinate_last, float wavenumber_first, float wavenumber_last) {
    return MAX(
           MAX(coordinate_first * wavenumber_first, coordinate_first * wavenumber_last),
           MAX(coordinate_last  * wavenumber_first, coordinate_last  * wavenumber_last));
}

void init_metadata(void *ptr, void *_uvw, void *_wavenumbers, int nr_stations, int nr_baselines, int nr_timesteps, int nr_timeslots, int nr_channels, int gridsize, int subgridsize, float imagesize) {
    // Compute number of subgrids
    int nr_subgrids = nr_baselines * nr_timeslots;

    // nr_time is the total number of timesteps for a baseline
    int nr_time = nr_timesteps * nr_timeslots;

    // Define datatypes
    TYPEDEF_UVW
    TYPEDEF_UVW_TYPE
    TYPEDEF_WAVENUMBER_TYPE
    TYPEDEF_BASELINE
    TYPEDEF_BASELINE_TYPE
    TYPEDEF_COORDINATE
    TYPEDEF_METADATA
    TYPEDEF_METADATA_TYPE

    // Pointers to datastructures
    UVWType *uvw = (UVWType *) _uvw;
    WavenumberType *wavenumbers = (WavenumberType *) _wavenumbers;
    BaselineType *baselines = (BaselineType *) init_baselines(nr_stations, nr_baselines);
    MetadataType *metadata = (MetadataType *) ptr;

    // Get wavenumber for first and last frequency
    float wavenumber_first = (*wavenumbers)[0];
    float wavenumber_last  = (*wavenumbers)[nr_channels-1];

    // Iterate all baselines
    for (int bl = 0; bl < nr_baselines; bl++) {
        // Load baseline
        Baseline baseline = (*baselines)[bl];

        // Iterate all timeslots
        for (int timeslot = 0; timeslot < nr_timeslots; timeslot++) {
            int time_offset = timeslot * nr_timesteps;

            // Find mininmum and maximum u and v for current timeslot in pixels
            float u_min =  std::numeric_limits<float>::infinity();
            float u_max = -std::numeric_limits<float>::infinity();
            float v_min =  std::numeric_limits<float>::infinity();
            float v_max = -std::numeric_limits<float>::infinity();

            // Iterate all timesteps
            for (int timestep = 0; timestep < nr_timesteps; timestep++) {
                UVW current = (*uvw)[bl][time_offset + timestep];

                // U,V in meters
                float u_meters = current.u;
                float v_meters = current.v;

                // Iterate all channels
                for (int chan = 0; chan < nr_channels; chan++) {
                    float wavenumber = (*wavenumbers)[chan];
                    float scaling = imagesize * wavenumber / (2 * M_PI);

                    // U,V in pixels
                    float u_pixels = u_meters * scaling;
                    float v_pixels = v_meters * scaling;

                    if (u_pixels < u_min) u_min = u_pixels;
                    if (u_pixels > u_max) u_max = u_pixels;
                    if (v_pixels < v_min) v_min = v_pixels;
                    if (v_pixels > v_max) v_max = v_pixels;
                }
            }

            // Compute middle point in pixels
            int u_pixels = roundf((u_max + u_min) / 2);
            int v_pixels = roundf((v_max + v_min) / 2);

            // Shift center from middle of grid to top left
            u_pixels += (gridsize/2);
            v_pixels += (gridsize/2);

            // Shift from middle of subgrid to top left
            u_pixels -= (subgridsize/2);
            v_pixels -= (subgridsize/2);

            // Construct coordinate
            Coordinate coordinate = { u_pixels, v_pixels };

            // Compute subgrid number
            int subgrid_nr = bl * nr_timeslots + timeslot;

            // Set metadata
            Metadata m = { timeslot, baseline, coordinate };
            (*metadata)[subgrid_nr] = m;
        }
    }

    // Free memory
    free(baselines);
}

/*
    Methods where memory is allocated
*/
void* init_uvw(int nr_stations, int nr_baselines, int nr_time) {
    TYPEDEF_UVW
    TYPEDEF_UVW_TYPE
    void *ptr = malloc(sizeof(UVWType));
    init_uvw(ptr, nr_stations, nr_baselines, nr_time);
    return ptr;
}

void* init_visibilities(int nr_baselines, int nr_time, int nr_channels, int nr_polarizations) {
    TYPEDEF_VISIBILITIES_TYPE
    void *ptr = malloc(sizeof(VisibilitiesType));
    init_visibilities(ptr, nr_baselines, nr_time, nr_channels, nr_polarizations);
    return ptr;
}

void* init_wavenumbers(int nr_channels) {
    TYPEDEF_WAVENUMBER_TYPE
    void *ptr = malloc(sizeof(WavenumberType));
    init_wavenumbers(ptr, nr_channels);
    return ptr;
}

void* init_aterm(int nr_stations, int nr_timeslots, int nr_polarizations,
                 int subgridsize) {
    TYPEDEF_ATERM_TYPE
    void *ptr = malloc(sizeof(ATermType));
    init_aterm(ptr, nr_stations, nr_timeslots, nr_polarizations, subgridsize);
    return ptr;
}

void* init_spheroidal(int subgridsize) {
    TYPEDEF_SPHEROIDAL_TYPE
    void *ptr = malloc(sizeof(SpheroidalType));
    init_spheroidal(ptr, subgridsize);
    return ptr;
}

void* init_baselines(int nr_stations, int nr_baselines) {
    TYPEDEF_BASELINE
    TYPEDEF_BASELINE_TYPE
    void *ptr = malloc(sizeof(BaselineType));
    init_baselines(ptr, nr_stations, nr_baselines);
    return ptr;

}

void* init_subgrid(int nr_baselines, int subgridsize, int nr_polarizations, int nr_chunks) {
    TYPEDEF_SUBGRID_TYPE
    void *ptr = malloc(sizeof(SubGridType));
    init_subgrid(ptr, nr_baselines, subgridsize, nr_polarizations, nr_chunks);
    return ptr;
}

void* init_grid(int gridsize, int nr_polarizations) {
    TYPEDEF_GRID_TYPE
    void *ptr = malloc(sizeof(GridType));
    init_grid(ptr, gridsize, nr_polarizations);
    return ptr;
}

void *init_metadata(void *uvw, void *wavenumbers, int nr_stations, int nr_baselines, int nr_timesteps, int nr_timeslots, int nr_channels, int gridsize, int subgridsize, float imagesize) {
   int nr_subgrids = nr_baselines * nr_timeslots;
   TYPEDEF_BASELINE
   TYPEDEF_COORDINATE
   TYPEDEF_METADATA
   TYPEDEF_METADATA_TYPE
   void *ptr = malloc(sizeof(MetadataType));
   init_metadata(ptr, uvw, wavenumbers, nr_stations, nr_baselines, nr_timesteps, nr_timeslots, nr_channels, gridsize, subgridsize, imagesize);
   return ptr;
}
} // namespace idg





// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    void utils_init_uvw(
         void *ptr,
         int nr_stations,
         int nr_baselines,
         int nr_time,
         int integration_time)
    {
        idg::init_uvw(ptr, nr_stations, nr_baselines, nr_time, integration_time);
    }

    void utils_init_wavenumbers(void *ptr, int nr_channels)
    {
         idg::init_wavenumbers(ptr, nr_channels);
    }

    void utils_init_metadata(
         void *ptr,
         void *uvw,
         void *wavenumbers,
         int nr_stations,
         int nr_baselines,
         int nr_timesteps,
         int nr_timeslots,
         int nr_channels,
         int gridsize,
         int subgridsize,
         float imagesize)
    {
         idg::init_metadata(ptr, uvw, wavenumbers, nr_stations, nr_baselines,
                            nr_timesteps, nr_timeslots, nr_channels, gridsize,
                            subgridsize, imagesize);
    }

    void utils_init_visibilities(
        void *ptr,
        int nr_baselines,
        int nr_time,
        int nr_channels,
        int nr_polarizations)
    {
        idg::init_visibilities(ptr, nr_baselines, nr_time,
                               nr_channels, nr_polarizations);
    }


    void utils_init_aterms(
        void *ptr,
        int nr_stations,
        int nr_timeslots,
        int nr_polarizations,
        int subgridsize)
    {
        idg::init_aterm(ptr, nr_stations, nr_timeslots,
                        nr_polarizations, subgridsize);
    }


    void utils_init_spheroidal(
        void *ptr,
        int subgridsize)
    {
        idg::init_spheroidal(ptr, subgridsize);
    }


}  // end extern "C"
