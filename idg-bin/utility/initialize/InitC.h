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
    idg::float2 visibility = {1.0f, 0.0f};

    // Set all visibilities
    #pragma omp parallel for collapse(4)
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


void init_identity_spheroidal(void *ptr, int subgridsize) {
    TYPEDEF_SPHEROIDAL_TYPE
    SpheroidalType *spheroidal = (SpheroidalType *) ptr;

    float value = 1.0;

    for (int y = 0; y < subgridsize; y++) {
        for (int x = 0; x < subgridsize; x++) {
             (*spheroidal)[y][x] = value;
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

    float x[subgridsize];
    for (int i = 0; i < subgridsize; i++) {
        float tmp = fabs(-1 + (i+0.5)*2.0f/float(subgridsize));
        x[i] = idg::evaluate_spheroidal(tmp);
    }

    for (int i = 0; i < subgridsize; i++) {
        for (int j = 0; j < subgridsize; j++) {
             (*spheroidal)[i][j] = x[i]*x[j];
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


extern "C" {

    void utils_init_identity_aterms(
        void *ptr,
        int nr_timeslots,
        int nr_stations,
        int subgridsize,
        int nr_polarizations)
    {
        init_identity_aterm(
            ptr, nr_timeslots, nr_stations,
            subgridsize, nr_polarizations);
    }


    void* utils_init_identity_spheroidal(void *ptr, int subgridsize)
    {
        init_identity_spheroidal(ptr, subgridsize);
    }


    void utils_init_example_uvw(
         void *ptr,
         int nr_stations,
         int nr_baselines,
         int nr_time,
         int integration_time)
    {
        idg::init_example_uvw(
            ptr, nr_stations, nr_baselines,
            nr_time, integration_time);
    }


    void utils_init_example_wavenumbers(void *ptr, int nr_channels)
    {
         init_example_wavenumbers(ptr, nr_channels);
    }

    void utils_init_example_visibilities(
        void *ptr,
        int nr_baselines,
        int nr_time,
        int nr_channels,
        int nr_polarizations)
    {
        init_example_visibilities(
            ptr, nr_baselines, nr_time,
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
        add_pt_src(
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
        init_example_aterm(
            ptr, nr_timeslots, nr_stations,
            subgridsize, nr_polarizations);
    }


    void utils_init_example_aterms_offset(
        void *ptr,
        int nr_timeslots,
        int nr_time)
    {
        init_example_aterm_offsets(
            ptr, nr_timeslots, nr_time);
    }


    void utils_init_example_spheroidal(
        void *ptr,
        int subgridsize)
    {
        init_example_spheroidal(ptr, subgridsize);
    }


    void utils_init_example_baselines(
        void *ptr,
        int nr_stations,
        int nr_baselines)
    {
        init_example_baselines(ptr, nr_stations, nr_baselines);
    }


    void utils_resize_spheroidal(
        float *spheroidal,
        int   subgrid_size,
        float *spheroidal_resized,
        int   size)
    {
        resize_spheroidal(
            spheroidal,
            subgrid_size,
            spheroidal_resized,
            size);
    }

}  // end extern "C"
