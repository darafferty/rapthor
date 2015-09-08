#pragma omp declare target

#include <complex>
#include <cmath>
#include <cstdio>
#include <immintrin.h>
#include <omp.h>
#include <cstring> // memset
#include <cstdint>

#include "Types.h"

#define NR_POLARIZATIONS 4

namespace idg {

void kernel_gridder (
	const int jobsize, const float w_offset,
	const void *_uvw,
	const void *_wavenumbers,
	const void *_visibilities,
	const void *_spheroidal,
	const void *_aterm,
	const void *_metadata,
	void	   *_subgrid,
    const int nr_stations,
    const int nr_timesteps,
    const int nr_timeslots,
    const int nr_channels,
    const int subgridsize,
    const float imagesize,
    const int nr_polarizations
	) {
    TYPEDEF_UVW
    TYPEDEF_UVW_TYPE
    TYPEDEF_WAVENUMBER_TYPE
    TYPEDEF_VISIBILITIES_TYPE
    TYPEDEF_SPHEROIDAL_TYPE
    TYPEDEF_ATERM_TYPE
    TYPEDEF_BASELINE
    TYPEDEF_COORDINATE
    TYPEDEF_METADATA
    TYPEDEF_METADATA_TYPE
    TYPEDEF_SUBGRID_TYPE

    UVWType *uvw = (UVWType *) _uvw;
    WavenumberType *wavenumbers = (WavenumberType *) _wavenumbers;
    VisibilitiesType *visibilities = (VisibilitiesType *) _visibilities;
    SpheroidalType *spheroidal = (SpheroidalType *) _spheroidal;
    ATermType *aterm = (ATermType *) _aterm;
    MetadataType *metadata = (MetadataType *) _metadata;
    SubGridType *subgrid = (SubGridType *) _subgrid;

    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    {
    // Iterate all subgrids
    #pragma omp for
	for (int s = 0; s < jobsize; s++) {
        // Load metadata
        const Metadata m = (*metadata)[s];
        int time_nr = m.time_nr;
        int station1 = m.baseline.station1;
        int station2 = m.baseline.station2;
        int x_coordinate = m.coordinate.x;
        int y_coordinate = m.coordinate.y;

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + subgridsize/2) / imagesize;
        float v_offset = (y_coordinate + subgridsize/2) / imagesize;

        // Initialize private subgrid
        FLOAT_COMPLEX pixels[subgridsize][subgridsize][NR_POLARIZATIONS] __attribute__((aligned(64)));
        memset(pixels, 0, subgridsize * subgridsize * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));

        // Storage for precomputed values
        float phase_index[subgridsize][subgridsize]  __attribute__((aligned(64)));
        float phase_offset[subgridsize][subgridsize] __attribute__((aligned(64)));
        FLOAT_COMPLEX vis[NR_POLARIZATIONS][nr_channels] __attribute__((aligned(64)));

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = (*uvw)[s][time].u;
            float v = (*uvw)[s][time].v;
            float w = (*uvw)[s][time].w;

            // Compute phase indices and phase offsets
            for (int y = 0; y < subgridsize; y++) {
                for (int x = 0; x < subgridsize; x++) {
                    // Compute l,m,n
                    float l = -(x-(subgridsize/2)) * imagesize/subgridsize;
                    float m =  (y-(subgridsize/2)) * imagesize/subgridsize;
                    float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                    // Compute phase index
                    phase_index[y][x] = u*l + v*m + w*n;

                    // Compute phase offset
                    phase_offset[y][x] = u_offset*l + v_offset*m + w_offset*n;
                }
            }

            // Preload visibilities
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                for (int chan = 0; chan < nr_channels; chan++) {
                    vis[pol][chan] = (*visibilities)[s][time][chan][pol];
                }
            }

            // Compute phasor and update current subgrid
            #pragma simd
            for (int chan = 0; chan < nr_channels; chan++) {
                for (int y = 0; y < subgridsize; y++) {
                    for (int x = 0; x < subgridsize; x++) {
                        float phase = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
                        FLOAT_COMPLEX phasor = FLOAT_COMPLEX(cosf(phase), sinf(phase));

                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            pixels[y][x][pol] += vis[pol][chan] * phasor;
                        }
                    }
                }
            }
        }

        // Apply aterm and spheroidal and store result
        for (int y = 0; y < subgridsize; y++) {
            for (int x = 0; x < subgridsize; x++) {
                // Load a term for station1
                FLOAT_COMPLEX aXX1 = (*aterm)[station1][time_nr][0][y][x];
                FLOAT_COMPLEX aXY1 = (*aterm)[station1][time_nr][1][y][x];
                FLOAT_COMPLEX aYX1 = (*aterm)[station1][time_nr][2][y][x];
                FLOAT_COMPLEX aYY1 = (*aterm)[station1][time_nr][3][y][x];

                // Load aterm for station2
                FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][time_nr][0][y][x]);
                FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][time_nr][1][y][x]);
                FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][time_nr][2][y][x]);
                FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][time_nr][3][y][x]);

                // Load uv values
                FLOAT_COMPLEX pixelsXX = pixels[y][x][0];
                FLOAT_COMPLEX pixelsXY = pixels[y][x][1];
                FLOAT_COMPLEX pixelsYX = pixels[y][x][2];
                FLOAT_COMPLEX pixelsYY = pixels[y][x][3];

                // Apply aterm to subgrid
                pixels[y][x][0]  = (pixelsXX * aXX1);
                pixels[y][x][0] += (pixelsXY * aYX1);
                pixels[y][x][0] += (pixelsXX * aXX2);
                pixels[y][x][0] += (pixelsYX * aYX2);
                pixels[y][x][1]  = (pixelsXX * aXY1);
                pixels[y][x][1] += (pixelsXY * aYY1);
                pixels[y][x][1] += (pixelsXY * aXX2);
                pixels[y][x][1] += (pixelsYY * aYX2);
                pixels[y][x][2]  = (pixelsYX * aXX1);
                pixels[y][x][2] += (pixelsYY * aYX1);
                pixels[y][x][2] += (pixelsXX * aXY2);
                pixels[y][x][2] += (pixelsYX * aYY2);
                pixels[y][x][3]  = (pixelsYX * aXY1);
                pixels[y][x][3] += (pixelsYY * aYY1);
                pixels[y][x][3] += (pixelsXY * aXY2);
                pixels[y][x][3] += (pixelsYY * aYY2);

                // Load spheroidal
                float sph = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_dst = (x + (subgridsize/2)) % subgridsize;
                int y_dst = (y + (subgridsize/2)) % subgridsize;

                // Set subgrid value
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    (*subgrid)[s][pol][y_dst][x_dst] = pixels[y][x][pol] * sph;
                }
            }
        }
    }
    }
}


uint64_t kernel_gridder_flops(int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations) {
    return
    1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (
        // Phasor
        2 * 22 +
        // UV
        NR_POLARIZATIONS * 8) +
    // ATerm
    1ULL * jobsize * subgridsize * subgridsize * NR_POLARIZATIONS * 30 +
    // Spheroidal
    1ULL * jobsize * subgridsize * subgridsize * NR_POLARIZATIONS * 2 +
    // Shift
    1ULL * jobsize * subgridsize * subgridsize * NR_POLARIZATIONS * 6;
}

uint64_t kernel_gridder_bytes(int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations) {
    return
    // Grid
    1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // ATerm
    1ULL * jobsize * subgridsize * subgridsize * (2 * sizeof(unsigned)) + (2 * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // Spheroidal
    1ULL * jobsize * subgridsize * subgridsize * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
}

} // end namespace idg

#pragma omp end declare target
