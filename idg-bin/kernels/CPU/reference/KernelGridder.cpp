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
        Metadata m = metadata[s];
        int time_nr = m.time_nr;
        int station1 = m.baseline.station1;
        int station2 = m.baseline.station2;
        int x_coordinate = m.coordinates.x;
        int y_coordinate = m.coordinates.y;
        
        // Compute u and v offset
        float u_offset = x_coordinate / IMAGESIZE;
        float v_offset = x_coordinate / IMAGESIZE;

        // Initialize private subgrid
        FLOAT_COMPLEX pixels[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
        memset(pixels, 0, SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));

        // Storage for precomputed values
        float phase_index[SUBGRIDSIZE][SUBGRIDSIZE]  __attribute__((aligned(32)));
        float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        FLOAT_COMPLEX visXX[NR_CHANNELS]             __attribute__((aligned(32)));
        FLOAT_COMPLEX visXY[NR_CHANNELS]             __attribute__((aligned(32)));
        FLOAT_COMPLEX visYX[NR_CHANNELS]             __attribute__((aligned(32)));
        FLOAT_COMPLEX visYY[NR_CHANNELS]             __attribute__((aligned(32)));
        float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));
        float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));
        
        // Iterate all timesteps
        for (int time = 0; time < NR_TIME; time++) {
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
                } }
        
            // Load visibilities
            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                visXX[chan] = (*visibilities)[s][time][chan][0];
                visXY[chan] = (*visibilities)[s][time][chan][1];
                visYX[chan] = (*visibilities)[s][time][chan][2];
                visYY[chan] = (*visibilities)[s][time][chan][3];
            }
        
            // Compute phasor
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                        // Compute phase
                        float wavenumber = (*wavenumbers)[chan];
                        float phase  = (phase_index[y][x] * wavenumber) - phase_offset[y][x];
                    
                        // Compute phasor
                        phasor_real[y][x][chan] = cosf(phase);
                        phasor_imag[y][x][chan] = sinf(phase);
                    }
                }
            }
    
            // Update current subgrid
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    FLOAT_COMPLEX phasor[NR_CHANNELS] __attribute__((aligned(32)));
                    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                        phasor[chan] = FLOAT_COMPLEX(phasor_real[y][x][chan], phasor_imag[y][x][chan]);
                    }
            
                    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                        pixels[y][x][0] += visXX[chan] * phasor[chan];
                        pixels[y][x][1] += visXY[chan] * phasor[chan];
                        pixels[y][x][2] += visYX[chan] * phasor[chan];
                        pixels[y][x][3] += visYY[chan] * phasor[chan];
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
                float s = (*spheroidal)[y][x];
                
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
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++ ) {
                    (*subgrid)[bl][chunk][pol][y_dst][x_dst] = pixels[y][x][pol] * s;
                }
            }
        }
    }
}

uint64_t kernel_gridder_flops(int jobsize) {
    return 
    1ULL * jobsize * NR_TIME * SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS * (
        // Phasor
        2 * 22 + 
        // UV
        NR_POLARIZATIONS * 8) +
    // ATerm
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 30 +
    // Spheroidal
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2 +
    // Shift
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 6;
}

uint64_t kernel_gridder_bytes(int jobsize) {
    return
    // Grid
    1ULL * jobsize * NR_TIME * SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS * (NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // ATerm
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * (2 * sizeof(unsigned)) + (2 * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // Spheroidal
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
}
}
