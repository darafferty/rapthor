#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"

extern "C" {
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

        // Iterate all pixels in subgrid
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Initialize pixel for every polarization
                FLOAT_COMPLEX pixels[NR_POLARIZATIONS];
                memset(pixels, 0, NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));

                // Compute l,m,n
                float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
 
                // Iterate all timesteps
                for (int time = 0; time < NR_TIMESTEPS; time++) {
                    // Load UVW coordinates
                    float u = (*uvw)[s][time].u;
                    float v = (*uvw)[s][time].v;
                    float w = (*uvw)[s][time].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phase offset
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    // Update pixel for every channel
                    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                        // Compute phase
                        float wavenumber = (*wavenumbers)[chan];
                        float phase  = (phase_index * wavenumber) - phase_offset;

                        // Compute phasor
                        float phasor_real = cosf(phase);
                        float phasor_imag = sinf(phase);
                        FLOAT_COMPLEX phasor = FLOAT_COMPLEX(phasor_real, phasor_imag);

                        // Update pixel for every polarization
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            FLOAT_COMPLEX visibility = (*visibilities)[s][time][chan][pol];
                            pixels[pol] += visibility * phasor;
                        }
                    }
                }
                
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
                FLOAT_COMPLEX pixelsXX = pixels[0];
                FLOAT_COMPLEX pixelsXY = pixels[1];
                FLOAT_COMPLEX pixelsYX = pixels[2];
                FLOAT_COMPLEX pixelsYY = pixels[3];

                // Apply aterm to subgrid
                pixels[0]  = (pixelsXX * aXX1);
                pixels[0] += (pixelsXY * aYX1);
                pixels[0] += (pixelsXX * aXX2);
                pixels[0] += (pixelsYX * aYX2);
                pixels[1]  = (pixelsXX * aXY1);
                pixels[1] += (pixelsXY * aYY1);
                pixels[1] += (pixelsXY * aXX2);
                pixels[1] += (pixelsYY * aYX2);
                pixels[2]  = (pixelsYX * aXX1);
                pixels[2] += (pixelsYY * aYX1);
                pixels[2] += (pixelsXX * aXY2);
                pixels[2] += (pixelsYX * aYY2);
                pixels[3]  = (pixelsYX * aXY1);
                pixels[3] += (pixelsYY * aYY1);
                pixels[3] += (pixelsXY * aXY2);
                pixels[3] += (pixelsYY * aYY2);

                // Load spheroidal
                float sph = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Set subgrid value
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    (*subgrid)[s][pol][y_dst][x_dst] = pixels[pol] * sph;
                }
            }
        }
    }
    }
}
}
