#include "Init.h"

#define TYPEDEF_UVW               typedef struct { float u, v, w; } UVW;
#define TYPEDEF_UVW_TYPE          typedef UVW UVWType[nr_baselines][nr_time];
#define TYPEDEF_VISIBILITIES_TYPE typedef std::complex<float> VisibilitiesType[nr_baselines][nr_time][nr_channels][nr_polarizations];
#define TYPEDEF_WAVENUMBER_TYPE   typedef float WavenumberType[nr_channels];
#define TYPEDEF_ATERM_TYPE        typedef std::complex<float> ATermType[nr_stations][nr_polarizations][subgridsize][subgridsize];
#define TYPEDEF_SPHEROIDAL_TYPE   typedef float SpheroidalType[subgridsize][subgridsize];
#define TYPEDEF_BASELINE          typedef struct { int station1, station2; } Baseline;
#define TYPEDEF_BASELINE_TYPE     typedef Baseline BaselineType[nr_baselines];
#define TYPEDEF_SUBGRID_TYPE      typedef std::complex<float> SubGridType[nr_baselines][nr_chunks][subgridsize][subgridsize][nr_polarizations];
#define TYPEDEF_GRID_TYPE         typedef std::complex<float> GridType[nr_polarizations][gridsize][gridsize];


namespace idg {

/*
    Methos where pointed to allocated memory is provided
*/
void init_uvw(void *ptr, int nr_stations, int nr_baselines, int nr_time, int gridsize, int subgridsize, int w_planes) {
    TYPEDEF_UVW
    TYPEDEF_UVW_TYPE
    
    UVWType *uvw = (UVWType *) ptr;
    
    // Check whether layout file exists
    char* filename;
    bool found = false;
    for (int i = 0; i < 4; i++) {
        std::stringstream ss;
        for (int j = 0; j < i; j++) {
            ss << "../";
        }
        ss << LAYOUT_FILE;
        filename = (char *) ss.str().c_str();

        if (uvwsim_file_exists(filename)) {
            found = true;
            break;    
        }
    }

    if (!found) {
        std::cerr << "Unable to find specified layout file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Read the number of stations in the layout file.
    int nr_stations_file = uvwsim_get_num_stations(filename);

    // Check wheter the requested number of station is feasible
    if (nr_stations_file < nr_stations) {
        std::cerr << "More stations requested than present in layout file: " 
                  << "(" << nr_stations_file << ")" << std::endl;
    }

    // Allocate memory for antenna coordinates
    double *x = (double*) malloc(nr_stations_file * sizeof(double));
    double *y = (double*) malloc(nr_stations_file * sizeof(double));
    double *z = (double*) malloc(nr_stations_file * sizeof(double));
    
    // Load the antenna coordinates
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
            int index = nr_stations * ((double) random() / RAND_MAX);
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
    double obs_length_days = 1.0 / 24.0;

    // Allocate memory for baseline coordinates
    int nr_coordinates = nr_time * nr_baselines;
    double *uu = (double*) malloc(nr_coordinates * sizeof(double));
    double *vv = (double*) malloc(nr_coordinates * sizeof(double));
    double *ww = (double*) malloc(nr_coordinates * sizeof(double));

    // Evaluate baseline uvw coordinates.
    for (int t = 0; t < nr_time; t++) {
        double time_mjd = start_time_mjd + t * (obs_length_days/(double)nr_time);
        size_t offset = t * nr_baselines;
        uvwsim_evaluate_baseline_uvw(
            &uu[offset], &vv[offset], &ww[offset],
            nr_stations, x, y, z, ra0, dec0, time_mjd);
    }
   
    // Find minimum and maxmium u, v and w values
    UVW min = {0, 0, 0};
    UVW max = {0, 0, 0};
    for (int bl = 0; bl < nr_baselines; bl++) {
        for (int t = 0; t < nr_time; t++) {
            int i = bl * nr_time + t;
            UVW value = {(float) uu[i], (float) vv[i], (float) ww[i]};
            if (value.u < min.u) min.u = value.u;
            if (value.v < min.v) min.v = value.v;
            if (value.w < min.w) min.w = value.w;
            if (value.u > max.u) max.u = value.u;
            if (value.v > max.v) max.v = value.v;
            if (value.w > max.w) max.w = value.w;
        }
    }

    // Scale uvw to to grid and collapse into w-planes
    float scale_u = ((gridsize/2.0f)-subgridsize-1) / MAX(abs(min.u), max.u);
    float scale_v = ((gridsize/2.0f)-subgridsize-1) / MAX(abs(min.v), max.v);
    float scale_w = (w_planes/2) / MAX(abs(min.w), max.w);

    for (int bl = 0; bl < nr_baselines; bl++) {
        for (int t = 0; t < nr_time; t++) {
            UVW value = (*uvw)[bl][t];
            value.u = scale_u * value.u + (gridsize/2.0f);
            value.v = scale_v * value.v + (gridsize/2.0f);
            value.w = scale_w * value.w + (w_planes/2.0f);
            (*uvw)[bl][t] = value;
        }
    }
  
    // Free memory 
    free(x); free(y); free(z);
    free(uu); free(vv); free(ww);
}

void init_visibilities(void *ptr, int nr_baselines, int nr_time, int nr_channels, int nr_polarizations) {
    TYPEDEF_VISIBILITIES_TYPE
	VisibilitiesType *visibilities = (VisibilitiesType *) (ptr);
	
	// Fixed visibility
	std::complex<float> visibility(2, 1);
	
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

void init_aterm(void *ptr, int nr_stations, int nr_polarizations, int subgridsize) {
	TYPEDEF_ATERM_TYPE
    ATermType *aterm = (ATermType *) ptr;
	
	std::complex<float> value(1, 1);
	
	for (int ant = 0; ant < nr_stations; ant++) {
		for (int y = 0; y < subgridsize; y++) {
			for (int x = 0; x < subgridsize; x++) {
				for (int pol = 0; pol < nr_polarizations; pol++) {
					(*aterm)[ant][pol][y][x] = value;
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


/*
    Methods where memory is allocated
*/
void* init_uvw(int nr_stations, int nr_baselines, int nr_time, int gridsize, int subgridsize, int w_planes) {
    TYPEDEF_UVW
    TYPEDEF_UVW_TYPE
    void *ptr = malloc(sizeof(UVWType));
    init_uvw(ptr, nr_stations, nr_baselines, nr_time, gridsize, subgridsize, w_planes);
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

void* init_aterm(int nr_stations, int nr_polarizations, int subgridsize) {
    TYPEDEF_ATERM_TYPE
    void *ptr = malloc(sizeof(ATermType));
    init_aterm(ptr, nr_stations, nr_polarizations, subgridsize);
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

} // namespace idg
