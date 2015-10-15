#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"

extern "C" {
void kernel_degridder(
    const int jobsize, const float w_offset,
	const UVWType		 __restrict__ *uvw,
	const WavenumberType __restrict__ *wavenumbers,
	VisibilitiesType	 __restrict__ *visibilities,
	const SpheroidalType __restrict__ *spheroidal,
	const ATermType		 __restrict__ *aterm,
	const MetadataType	 __restrict__ *metadata,
	const SubGridType	 __restrict__ *subgrid
	) {
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
        FLOAT_COMPLEX _pixels[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS] __attribute__((aligned(32)));
        float phasor_real[NR_CHANNELS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float phasor_imag[NR_CHANNELS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float phase_index[SUBGRIDSIZE][SUBGRIDSIZE]  __attribute__((aligned(32)));
        float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2) / IMAGESIZE;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2) / IMAGESIZE;

        // Apply aterm to subgrid
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
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
                int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Load uv values
                FLOAT_COMPLEX pixelsXX = _spheroidal * (*subgrid)[s][0][y_src][x_src];
                FLOAT_COMPLEX pixelsXY = _spheroidal * (*subgrid)[s][1][y_src][x_src];
                FLOAT_COMPLEX pixelsYX = _spheroidal * (*subgrid)[s][2][y_src][x_src];
                FLOAT_COMPLEX pixelsYY = _spheroidal * (*subgrid)[s][3][y_src][x_src];

                // Apply aterm to subgrid
                _pixels[y][x][0]  = pixelsXX * aXX1;
                _pixels[y][x][0] += pixelsXY * aYX1;
                _pixels[y][x][0] += pixelsXX * aXX2;
                _pixels[y][x][0] += pixelsYX * aYX2;
                _pixels[y][x][1]  = pixelsXX * aXY1;
                _pixels[y][x][1] += pixelsXY * aYY1;
                _pixels[y][x][1] += pixelsXY * aXX2;
                _pixels[y][x][1] += pixelsYY * aYX2;
                _pixels[y][x][2]  = pixelsYX * aXX1;
                _pixels[y][x][2] += pixelsYY * aYX1;
                _pixels[y][x][2] += pixelsXX * aXY2;
                _pixels[y][x][2] += pixelsYX * aYY2;
                _pixels[y][x][3]  = pixelsYX * aXY1;
                _pixels[y][x][3] += pixelsYY * aYY1;
                _pixels[y][x][3] += pixelsXY * aXY2;
                _pixels[y][x][3] += pixelsYY * aYY2;
            }
        }

        // Iterate all timesteps
        for (int time = 0; time < NR_TIMESTEPS; time++) {
            // Load UVW coordinates
            float u = (*uvw)[s][time].u;
            float v = (*uvw)[s][time].v;
            float w = (*uvw)[s][time].w;

            // Compute phase indices and phase offsets
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute l,m,n
                    float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                    // Compute phase index
                    phase_index[y][x] = u*l + v*m + w*n;

                    // Compute phase offset
                    phase_offset[y][x] = u_offset*l + v_offset*m + w_offset*n;
                }
            }

            // Compute phasor
            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Compute phase
                        float wavenumber = (*wavenumbers)[chan];
                        float phase  = (phase_index[y][x] * wavenumber) - phase_offset[y][x];

                        // Compute phasor
                        phasor_real[chan][y][x] = cosf(phase);
                        phasor_imag[chan][y][x] = sinf(phase);
                    }
                }
            }

            FLOAT_COMPLEX sum[NR_POLARIZATIONS] __attribute__((aligned(32)));

            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                memset(sum, 0, NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));

                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        FLOAT_COMPLEX phasor = FLOAT_COMPLEX(phasor_real[chan][y][x], phasor_imag[chan][y][x]);

                        // Update all polarizations
                        sum[0] += _pixels[y][x][0] * phasor;
                        sum[1] += _pixels[y][x][1] * phasor;
                        sum[2] += _pixels[y][x][2] * phasor;
                        sum[3] += _pixels[y][x][3] * phasor;
                    }
                }

                // Set visibilities
                (*visibilities)[s][time][chan][0] = sum[0];
                (*visibilities)[s][time][chan][1] = sum[1];
                (*visibilities)[s][time][chan][2] = sum[2];
                (*visibilities)[s][time][chan][3] = sum[3];
            }
        }
	}
    }
}
}
