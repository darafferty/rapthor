#pragma omp declare target

#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"

#define NR_POLARIZATIONS 4

namespace idg {

void kernel_degridder(
    const int jobsize, const float w_offset,
	const void *_uvw,
	const void *_wavenumbers,
	      void *_visibilities,
	const void *_spheroidal,
	const void *_aterm,
	const void *_metadata,
	const void *_subgrid,
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

        // Storage for precomputed values
        FLOAT_COMPLEX _pixels[NR_POLARIZATIONS][subgridsize][subgridsize] __attribute__((aligned(64)));
        float phasor_real[nr_channels][subgridsize][subgridsize] __attribute__((aligned(64)));
        float phasor_imag[nr_channels][subgridsize][subgridsize] __attribute__((aligned(64)));
        float phase_index[subgridsize][subgridsize]  __attribute__((aligned(64)));
        float phase_offset[subgridsize][subgridsize] __attribute__((aligned(64)));

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + subgridsize/2) / imagesize;
        float v_offset = (y_coordinate + subgridsize/2) / imagesize;

        // Apply aterm to subgrid
        for (int y = 0; y < subgridsize; y++) {
            for (int x = 0; x < subgridsize; x++) {
                // Load aterm for station1
                FLOAT_COMPLEX aXX1 = (*aterm)[station1][time_nr][0][y][x];
                FLOAT_COMPLEX aXY1 = (*aterm)[station1][time_nr][1][y][x];
                FLOAT_COMPLEX aYX1 = (*aterm)[station1][time_nr][2][y][x];
                FLOAT_COMPLEX aYY1 = (*aterm)[station1][time_nr][3][y][x];

                // Load aterm for station2
                FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][time_nr][0][y][x]);
                FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][time_nr][1][y][x]);
                FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][time_nr][2][y][x]);
                FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][time_nr][3][y][x]);

                // Load spheroidal
                float _spheroidal = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_src = (x + (subgridsize/2)) % subgridsize;
                int y_src = (y + (subgridsize/2)) % subgridsize;

                // Load uv values
                FLOAT_COMPLEX pixelsXX = _spheroidal * (*subgrid)[s][0][y_src][x_src];
                FLOAT_COMPLEX pixelsXY = _spheroidal * (*subgrid)[s][1][y_src][x_src];
                FLOAT_COMPLEX pixelsYX = _spheroidal * (*subgrid)[s][2][y_src][x_src];
                FLOAT_COMPLEX pixelsYY = _spheroidal * (*subgrid)[s][3][y_src][x_src];

                // Apply aterm to subgrid
                _pixels[0][y][x]  = pixelsXX * aXX1;
                _pixels[0][y][x] += pixelsXY * aYX1;
                _pixels[0][y][x] += pixelsXX * aXX2;
                _pixels[0][y][x] += pixelsYX * aYX2;
                _pixels[1][y][x]  = pixelsXX * aXY1;
                _pixels[1][y][x] += pixelsXY * aYY1;
                _pixels[1][y][x] += pixelsXY * aXX2;
                _pixels[1][y][x] += pixelsYY * aYX2;
                _pixels[2][y][x]  = pixelsYX * aXX1;
                _pixels[2][y][x] += pixelsYY * aYX1;
                _pixels[2][y][x] += pixelsXX * aXY2;
                _pixels[2][y][x] += pixelsYX * aYY2;
                _pixels[3][y][x]  = pixelsYX * aXY1;
                _pixels[3][y][x] += pixelsYY * aYY1;
                _pixels[3][y][x] += pixelsXY * aXY2;
                _pixels[3][y][x] += pixelsYY * aYY2;
            }
        }

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

            // Compute phasor
            for (int y = 0; y < subgridsize; y++) {
                for (int chan = 0; chan < nr_channels; chan++) {
                    for (int x = 0; x < subgridsize; x++) {
                        // Compute phase
                        float wavenumber = (*wavenumbers)[chan];
                        float phase  = (phase_index[y][x] * wavenumber) - phase_offset[y][x];

                        // Compute phasor
                        phasor_real[chan][y][x] = cosf(phase);
                        phasor_imag[chan][y][x] = sinf(phase);
                    }
                }
            }

            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                for (int chan = 0; chan < nr_channels; chan++) {
                    FLOAT_COMPLEX sum;
                    for (int y = 0; y < subgridsize; y++) {
                        for (int x = 0; x < subgridsize; x++) {
                            FLOAT_COMPLEX phasor = FLOAT_COMPLEX(phasor_real[chan][y][x], phasor_imag[chan][y][x]);
                            sum += _pixels[pol][y][x] * phasor;
                        }
                    }
                    (*visibilities)[s][time][chan][pol] = sum;
                }
            }
        }
	}
    }
}
} // end namespace idg

#pragma omp end declare target
