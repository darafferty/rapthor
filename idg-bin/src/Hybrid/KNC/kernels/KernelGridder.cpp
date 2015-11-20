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
#define UPDATE_1 \
    const int subgridsize, \
    const int nr_channels, \
    const float wavenumbers[nr_channels], \
    const FLOAT_COMPLEX vis[NR_POLARIZATIONS][nr_channels], \
    const float phase_index[subgridsize][subgridsize], \
    const float phase_offset[subgridsize][subgridsize], \
    FLOAT_COMPLEX pixels[subgridsize][subgridsize][NR_POLARIZATIONS]) {

#define UPDATE_2 \
        for (int y = 0; y < subgridsize; y++) { \
            for (int x = 0; x < subgridsize; x++) { \
                float phase = (phase_index[y][x] * wavenumbers[chan]) - phase_offset[y][x]; \
                FLOAT_COMPLEX phasor = FLOAT_COMPLEX(cosf(phase), sinf(phase)); \
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) { \
                    pixels[y][x][pol] += vis[pol][chan] * phasor; \
                } \
            } \
        } \
    }

namespace idg {

void update_4(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 4; chan++) {
    UPDATE_2
}

void update_8(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 8; chan++) {
    UPDATE_2
}

void update_9(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 9; chan++) {
    UPDATE_2
}

void update_10(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 10; chan++) {
    UPDATE_2
}

void update_11(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 11; chan++) {
    UPDATE_2
}

void update_12(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 12; chan++) {
    UPDATE_2
}

void update_13(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 13; chan++) {
    UPDATE_2
}

void update_14(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 14; chan++) {
    UPDATE_2
}

void update_15(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 15; chan++) {
    UPDATE_2
}

void update_16(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 16; chan++) {
    UPDATE_2
}

void update_17(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 17; chan++) {
    UPDATE_2
}

void update_18(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 18; chan++) {
    UPDATE_2
}

void update_19(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 19; chan++) {
    UPDATE_2
}

void update_20(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 20; chan++) {
    UPDATE_2
}

void update_21(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 21; chan++) {
    UPDATE_2
}

void update_22(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 22; chan++) {
    UPDATE_2
}

void update_23(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 23; chan++) {
    UPDATE_2
}

void update_24(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 24; chan++) {
    UPDATE_2
}

void update_25(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 25; chan++) {
    UPDATE_2
}

void update_26(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 26; chan++) {
    UPDATE_2
}

void update_27(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 27; chan++) {
    UPDATE_2
}

void update_28(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 28; chan++) {
    UPDATE_2
}

void update_29(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 29; chan++) {
    UPDATE_2
}

void update_30(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 30; chan++) {
    UPDATE_2
}

void update_31(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 31; chan++) {
    UPDATE_2
}

void update_32(
    UPDATE_1
    #pragma simd
    for (int chan = 0; chan < 32; chan++) {
    UPDATE_2
}

void update_n(
    const int subgridsize,
    const int nr_channels,
    const float wavenumbers[nr_channels],
    const FLOAT_COMPLEX vis[NR_POLARIZATIONS][nr_channels],
    const float phase_index[subgridsize][subgridsize],
    const float phase_offset[subgridsize][subgridsize],
    FLOAT_COMPLEX pixels[subgridsize][subgridsize][NR_POLARIZATIONS]
    ) {
    #pragma simd
    for (int chan = 0; chan < nr_channels; chan++) {
        for (int y = 0; y < subgridsize; y++) {
            for (int x = 0; x < subgridsize; x++) {
                float phase = (phase_index[y][x] * wavenumbers[chan]) - phase_offset[y][x];
                FLOAT_COMPLEX phasor = FLOAT_COMPLEX(cosf(phase), sinf(phase));

                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    pixels[y][x][pol] += vis[pol][chan] * phasor;
                }
            }
        }
    }
}

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
            #define UPDATE_PARAMETERS subgridsize, nr_channels, *wavenumbers, vis, phase_index, phase_offset, pixels
            if (nr_channels ==  4) { update_4(UPDATE_PARAMETERS);  continue; }
            if (nr_channels ==  8) { update_8(UPDATE_PARAMETERS);  continue; }
            if (nr_channels ==  9) { update_9(UPDATE_PARAMETERS);  continue; }
            if (nr_channels == 10) { update_10(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 11) { update_11(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 12) { update_12(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 13) { update_13(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 14) { update_14(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 15) { update_15(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 16) { update_16(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 17) { update_17(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 18) { update_18(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 19) { update_19(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 20) { update_20(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 21) { update_21(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 22) { update_22(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 23) { update_23(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 24) { update_24(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 25) { update_25(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 26) { update_26(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 27) { update_27(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 28) { update_28(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 29) { update_29(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 30) { update_30(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 31) { update_31(UPDATE_PARAMETERS); continue; }
            if (nr_channels == 32) { update_31(UPDATE_PARAMETERS); continue; }
            update_n(subgridsize, nr_channels, *wavenumbers, vis, phase_index, phase_offset, pixels);
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
} // end namespace idg

#pragma omp end declare target
