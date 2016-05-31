#include "Init.h"

#define TYPEDEF_UVW               typedef struct { float u, v, w; } UVW;
#define TYPEDEF_UVW_TYPE          typedef UVW UVWType[nr_baselines][nr_time];
#define TYPEDEF_VISIBILITIES_TYPE typedef idg::float2 VisibilitiesType[nr_baselines][nr_time][nr_channels][nr_polarizations];
#define TYPEDEF_WAVENUMBER_TYPE   typedef float WavenumberType[nr_channels];
#define TYPEDEF_ATERM_TYPE        typedef idg::float2 ATermType[nr_timeslots][nr_stations][subgridsize][subgridsize][nr_polarizations];
#define TYPEDEF_ATERM_OFFSET_TYPE typedef int ATermOffsetType[nr_timeslots + 1];
#define TYPEDEF_SPHEROIDAL_TYPE   typedef float SpheroidalType[subgridsize][subgridsize];
#define TYPEDEF_BASELINE          typedef struct { int station1, station2; } Baseline;
#define TYPEDEF_BASELINE_TYPE     typedef Baseline BaselineType[nr_baselines];
#define TYPEDEF_SUBGRID_TYPE      typedef idg::float2 SubGridType[nr_baselines][nr_chunks][subgridsize][subgridsize][nr_polarizations];
#define TYPEDEF_GRID_TYPE         typedef idg::float2 GridType[nr_polarizations][gridsize][gridsize];
#define TYPEDEF_COORDINATE        typedef struct { int x, y; } Coordinate;
#define TYPEDEF_METADATA          typedef struct { int time_nr; Baseline baseline; Coordinate coordinate; } Metadata;
#define TYPEDEF_METADATA_TYPE     typedef Metadata MetadataType[nr_subgrids];


namespace idg {

    const std::string ENV_LAYOUT_FILE  = "LAYOUT_FILE";

    /* Methods where pointed to allocated memory is provided */

    void init_example_uvw(
        void *ptr,
        int nr_stations,
        int nr_baselines,
        int nr_time,
        float integration_time)
    {
        TYPEDEF_UVW
        TYPEDEF_UVW_TYPE

        UVWType *uvw = (UVWType *) ptr;

        // Try to load layout file from environment
        char *cstr_layout_file = getenv(ENV_LAYOUT_FILE.c_str());

        // Check whether layout file exists
        char filename[512];
        if (cstr_layout_file) {
            sprintf(filename, "%s/%s/%s", IDG_SOURCE_DIR, LAYOUT_DIR, cstr_layout_file);
        } else {
            sprintf(filename, "%s/%s/%s", IDG_SOURCE_DIR, LAYOUT_DIR, LAYOUT_FILE);
        }

        //if (!uvwsim_file_exists(filename)) {
        //    std::cerr << "Unable to find specified layout file: "
        //              << filename << std::endl;
        //    exit(EXIT_FAILURE);
        //}

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


    void init_example_visibilities(
        void *ptr,
        int nr_baselines,
        int nr_time,
        int nr_channels,
        int nr_polarizations)
    {
        TYPEDEF_VISIBILITIES_TYPE
        VisibilitiesType *visibilities = (VisibilitiesType *) (ptr);

        // Fixed visibility
        // std::complex<float> visibility(1, 0);
        idg::float2 visibility = {1.0f, 0.0f};

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


    void add_pt_src(
        float x,
        float y,
        float amplitude,
        int   nr_baselines,
        int   nr_time,
        int   nr_channels,
        int   nr_polarizations,
        float imagesize,
        int   gridsize,
        void *uvw_,
        void *wavenumbers_,
        void *visibilities_)
    {
        TYPEDEF_UVW
        TYPEDEF_UVW_TYPE
        TYPEDEF_WAVENUMBER_TYPE
        TYPEDEF_VISIBILITIES_TYPE

        UVWType *uvw = (UVWType *) uvw_;
        WavenumberType *wavenumbers = (WavenumberType *) wavenumbers_;
        VisibilitiesType *visibilities = (VisibilitiesType *) visibilities_;

        float l = x * imagesize/gridsize;
        float m = y * imagesize/gridsize;

        #pragma omp parallel for
        for (int b = 0; b < nr_baselines; b++) {
            for (int t = 0; t < nr_time; t++) {
                for (int c = 0; c < nr_channels; c++) {
                    float u = (*wavenumbers)[c] * (*uvw)[b][t].u / (2 * M_PI);
                    float v = (*wavenumbers)[c] * (*uvw)[b][t].v / (2 * M_PI);
                    std::complex<float> value = amplitude *
                        std::exp(std::complex<float>(0, -2 * M_PI * (u*l + v*m)));
                    idg::float2 tmp = {value.real(), value.imag()};
                    for (int p = 0; p < nr_polarizations; p++) {
                        (*visibilities)[b][t][c][p] += tmp;
                    }
                }
            }
        }
    }


    void init_example_wavenumbers(
        void *ptr,
        int nr_channels)
    {
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


    void init_example_aterm(
        void *ptr,
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        TYPEDEF_ATERM_TYPE
        ATermType *aterm = (ATermType *) ptr;

        for (int t = 0; t < nr_timeslots; t++) {
            for (int ant = 0; ant < nr_stations; ant++) {
                for (int y = 0; y < subgridsize; y++) {
                    for (int x = 0; x < subgridsize; x++) {
                        (*aterm)[t][ant][y][x][0] = {1, 0};
                        (*aterm)[t][ant][y][x][1] = {0, 0};
                        (*aterm)[t][ant][y][x][2] = {0, 0};
                        (*aterm)[t][ant][y][x][3] = {1, 0};
                    }
                }
            }
        }
    }


    void init_example_aterm_offsets(
        void *ptr,
        int nr_timeslots,
        int nr_time)
    {
        TYPEDEF_ATERM_OFFSET_TYPE
        ATermOffsetType *aterm_offsets = (ATermOffsetType *) ptr;
        for (int time = 0; time < nr_timeslots; time++) {
             (*aterm_offsets)[time] = time * (nr_time / nr_timeslots);
        }
        (*aterm_offsets)[nr_timeslots] = nr_time;
    }


    void init_example_spheroidal(void *ptr, int subgridsize) {
        TYPEDEF_SPHEROIDAL_TYPE
        SpheroidalType *spheroidal = (SpheroidalType *) ptr;

        float value = 1.0;

        for (int y = 0; y < subgridsize; y++) {
            for (int x = 0; x < subgridsize; x++) {
                 (*spheroidal)[y][x] = value;
            }
        }
    }


    void init_example_baselines(
        void *ptr,
        int nr_stations,
        int nr_baselines)
    {
        TYPEDEF_BASELINE
        TYPEDEF_BASELINE_TYPE
        BaselineType *baselines = (BaselineType *) ptr;

        int bl = 0;

        for (int station1 = 0 ; station1 < nr_stations; station1++) {
            for (int station2 = station1 + 1; station2 < nr_stations; station2++) {
                if (bl >= nr_baselines) {
                    break;
                }
                (*baselines)[bl].station1 = station1;
                (*baselines)[bl].station2 = station2;
                bl++;
            }
        }
    }


    void init_example_subgrid(
        void *ptr,
        int nr_baselines,
        int subgridsize,
        int nr_polarizations,
        int nr_chunks)
    {
        TYPEDEF_SUBGRID_TYPE
        SubGridType *subgrid = (SubGridType *) ptr;
        memset(subgrid, 0, sizeof(SubGridType));
    }


    void init_example_grid(
        void *ptr,
        int gridsize,
        int nr_polarizations)
    {
        TYPEDEF_GRID_TYPE
        GridType *grid = (GridType *) ptr;
        memset(grid, 0, sizeof(GridType));
    }


    void init_zero_grid(
        void *ptr,
        int gridsize,
        int nr_polarizations)
    {
        TYPEDEF_GRID_TYPE
        GridType *grid = (GridType *) ptr;
        memset(grid, 0, sizeof(GridType));
    }


    void init_identity_aterm(
        void *ptr,
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        TYPEDEF_ATERM_TYPE
        ATermType *aterm = (ATermType *) ptr;

        for (int t = 0; t < nr_timeslots; t++) {
            for (int ant = 0; ant < nr_stations; ant++) {
                for (int y = 0; y < subgridsize; y++) {
                    for (int x = 0; x < subgridsize; x++) {
                        (*aterm)[t][ant][y][x][0] = {1, 0};
                        (*aterm)[t][ant][y][x][1] = {0, 0};
                        (*aterm)[t][ant][y][x][2] = {0, 0};
                        (*aterm)[t][ant][y][x][3] = {1, 0};
                    }
                }
            }
        }
    }


    /*
        Methods where memory is allocated
    */

    void* init_example_uvw(
        int nr_stations,
        int nr_baselines,
        int nr_time,
        float integration_time)
    {
        TYPEDEF_UVW
        TYPEDEF_UVW_TYPE
        void *ptr = malloc(sizeof(UVWType));
        init_example_uvw(ptr, nr_stations, nr_baselines, nr_time, integration_time);
        return ptr;
    }


    void* init_example_visibilities(
        int nr_baselines,
        int nr_time,
        int nr_channels,
        int nr_polarizations)
    {
        TYPEDEF_VISIBILITIES_TYPE
        void *ptr = malloc(sizeof(VisibilitiesType));
        init_example_visibilities(ptr, nr_baselines, nr_time, nr_channels, nr_polarizations);
        return ptr;
    }


    void* init_example_wavenumbers(int nr_channels)
    {
        TYPEDEF_WAVENUMBER_TYPE
        void *ptr = malloc(sizeof(WavenumberType));
        init_example_wavenumbers(ptr, nr_channels);
        return ptr;
    }


    void* init_example_aterm(
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        TYPEDEF_ATERM_TYPE
        void *ptr = malloc(sizeof(ATermType));
        init_example_aterm(ptr, nr_timeslots, nr_stations, subgridsize, nr_polarizations);
        return ptr;
    }


    void* init_example_aterm_offsets(int nr_timeslots, int nr_time)
    {
        TYPEDEF_ATERM_OFFSET_TYPE
        void *ptr = malloc(sizeof(ATermOffsetType));
        init_example_aterm_offsets(ptr, nr_timeslots, nr_time);
        return ptr;
    }


    void* init_example_spheroidal(int subgridsize)
    {
        TYPEDEF_SPHEROIDAL_TYPE
        void *ptr = malloc(sizeof(SpheroidalType));
        init_example_spheroidal(ptr, subgridsize);
        return ptr;
    }


    void* init_example_baselines(int nr_stations, int nr_baselines)
    {
        TYPEDEF_BASELINE
        TYPEDEF_BASELINE_TYPE
        void *ptr = malloc(sizeof(BaselineType));
        init_example_baselines(ptr, nr_stations, nr_baselines);
        return ptr;

    }


    void* init_example_subgrid(
        int nr_baselines,
        int subgridsize,
        int nr_polarizations,
        int nr_chunks)
    {
        TYPEDEF_SUBGRID_TYPE
        void *ptr = malloc(sizeof(SubGridType));
        init_example_subgrid(ptr, nr_baselines, subgridsize, nr_polarizations, nr_chunks);
        return ptr;
    }


    void* init_example_grid(int gridsize, int nr_polarizations)
    {
        TYPEDEF_GRID_TYPE
        void *ptr = malloc(sizeof(GridType));
        init_example_grid(ptr, gridsize, nr_polarizations);
        return ptr;
    }


    void* init_zero_grid(int gridsize, int nr_polarizations)
    {
        TYPEDEF_GRID_TYPE
        void *ptr = malloc(sizeof(GridType));
        init_zero_grid(ptr, gridsize, nr_polarizations);
        return ptr;
    }


    void* init_identity_aterm(
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        TYPEDEF_ATERM_TYPE
        void *ptr = malloc(sizeof(ATermType));
        init_identity_aterm(ptr, nr_timeslots, nr_stations, subgridsize, nr_polarizations);
        return ptr;
    }


} // namespace idg





// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    void utils_init_example_uvw(
         void *ptr,
         int nr_stations,
         int nr_baselines,
         int nr_time,
         int integration_time)
    {
        idg::init_example_uvw(ptr, nr_stations, nr_baselines, nr_time, integration_time);
    }

    void utils_init_example_wavenumbers(void *ptr, int nr_channels)
    {
         idg::init_example_wavenumbers(ptr, nr_channels);
    }

    void utils_init_example_visibilities(
        void *ptr,
        int nr_baselines,
        int nr_time,
        int nr_channels,
        int nr_polarizations)
    {
        idg::init_example_visibilities(ptr, nr_baselines, nr_time,
                               nr_channels, nr_polarizations);
    }

    void utils_add_pt_src(
        float x,
        float y,
        float amplitude,
        int nr_baselines,
        int nr_time,
        int nr_channels,
        int nr_polarizations,
        float imagesize,
        int gridsize,
        void *uvw,
        void *wavenumbers,
        void *visibilities)
    {
        idg::add_pt_src(
            x, y, amplitude, nr_baselines, nr_time, nr_channels,
            nr_polarizations, imagesize, gridsize,
            uvw, wavenumbers, visibilities);
    }

    void utils_init_example_aterms(
        void *ptr,
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        idg::init_example_aterm(ptr, nr_timeslots, nr_stations,
                        subgridsize, nr_polarizations);
    }

    void utils_init_example_aterms_offset(
        void *ptr,
        int nr_timeslots,
        int nr_time)
    {
        idg::init_example_aterm_offsets(ptr, nr_timeslots, nr_time);
    }

    void utils_init_example_spheroidal(
        void *ptr,
        int subgridsize)
    {
        idg::init_example_spheroidal(ptr, subgridsize);
    }

    void utils_init_example_baselines(
        void *ptr,
        int nr_stations,
        int nr_baselines)
    {
        idg::init_example_baselines(ptr, nr_stations, nr_baselines);
    }

    void utils_init_identity_aterms(
        void *ptr,
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        idg::init_identity_aterm(ptr, nr_timeslots, nr_stations,
                                 subgridsize, nr_polarizations);
    }


}  // end extern "C"
