#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>

#if defined(USING_INTEL_CXX_COMPILER)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"

extern "C" {
#if defined(USING_INTEL_CXX_COMPILER)
void kernel_gridder_intel(
	const int jobsize, const float w_offset,
	const UVWType		   __restrict__ *uvw,
	const WavenumberType   __restrict__ *wavenumbers,
	const VisibilitiesType __restrict__ *visibilities,
	const SpheroidalType   __restrict__ *spheroidal,
	const ATermType		   __restrict__ *aterm,
	const MetadataType	   __restrict__ *metadata,
	SubGridType			   __restrict__ *subgrid
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

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) / IMAGESIZE * 2 * M_PI;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) / IMAGESIZE * 2 * M_PI;

        // Initialize private subgrid
        FLOAT_COMPLEX pixels[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
        memset(pixels, 0, SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));

        // Storage for precomputed values
        float phase_index[SUBGRIDSIZE][SUBGRIDSIZE]  __attribute__((aligned(32)));
        float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        FLOAT_COMPLEX vis[NR_CHANNELS][NR_POLARIZATIONS] __attribute__((aligned(32)));
        float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));
        float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));
        float phase[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));

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
                    float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                    // Compute phase index
                    phase_index[y][x] = u*l + v*m + w*n;

                    // Compute phase offset
                    phase_offset[y][x] = u_offset*l + v_offset*m + w_offset*n;
                }
            }

            // Load visibilities
            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    vis[chan][pol] = (*visibilities)[s][time][chan][pol];
                }
            }

            // Compute phase
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                        phase[y][x][chan] = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
                    }
                }
            }

            // Compute phasor
            vmsSinCos(SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS,
                (const float *) &phase[0][0][0],
                                &phasor_imag[0][0][0],
                                &phasor_real[0][0][0], VML_PRECISION);

            // Update current subgrid
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                        FLOAT_COMPLEX phasor = FLOAT_COMPLEX(phasor_real[y][x][chan], phasor_imag[y][x][chan]);

                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            pixels[y][x][pol] += vis[chan][pol] * phasor;
                        }
                    }
                }
            }
       }

        // Apply aterm and spheroidal and store result
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
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

                // Load spheroidal
                float _spheroidal = (*spheroidal)[y][x];

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

                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Set subgrid value
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    (*subgrid)[s][pol][y_dst][x_dst] = pixels[y][x][pol] * _spheroidal;
                }
            }
        }
    }
    }
}
#endif

#if defined(USING_GNU_CXX_COMPILER)
void kernel_gridder_gnu(
	const int jobsize, const float w_offset,
	const UVWType		   __restrict__ *uvw,
	const WavenumberType   __restrict__ *wavenumbers,
	const VisibilitiesType __restrict__ *visibilities,
	const SpheroidalType   __restrict__ *spheroidal,
	const ATermType		   __restrict__ *aterm,
	const MetadataType	   __restrict__ *metadata,
	SubGridType			   __restrict__ *subgrid
	) {
    printf("%s not implemented yet\n", __func__);
}
#endif

void kernel_gridder(
	const int jobsize, const float w_offset,
	const UVWType		   __restrict__ *uvw,
	const WavenumberType   __restrict__ *wavenumbers,
	const VisibilitiesType __restrict__ *visibilities,
	const SpheroidalType   __restrict__ *spheroidal,
	const ATermType		   __restrict__ *aterm,
	const MetadataType	   __restrict__ *metadata,
	SubGridType			   __restrict__ *subgrid
	) {
    #if defined(USING_INTEL_CXX_COMPILER)
    kernel_gridder_intel(
          jobsize, w_offset, uvw, wavenumbers,
          visibilities, spheroidal, aterm, metadata, subgrid);
    #elif defined(USING_GNU_CXX_COMPILER)
    kernel_gridder_gnu(
          jobsize, w_offset, uvw, wavenumbers,
          visibilities, spheroidal, aterm, metadata, subgrid);
    #else 
    printf("%s not implemented yet, use Intel or GNU compiler\n", __func__);
    return;
    #endif
}

uint64_t kernel_gridder_flops(int jobsize) {
    return
    1ULL * jobsize * NR_TIMESTEPS * SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS * (
        // Phasor
        2 * 22 +
        // UV
        NR_POLARIZATIONS * 8) +
    // ATerm
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 30 +
    // Spheroidal
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2 +
    // Shift
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 6;
}

uint64_t kernel_gridder_bytes(int jobsize) {
    return
    // Grid
    1ULL * jobsize * NR_TIMESTEPS * SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS * (NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // ATerm
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (2 * sizeof(unsigned)) + (2 * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // Spheroidal
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
}
}
