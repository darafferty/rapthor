#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <likwid.h>
#include <stdint.h>

#define USE_VML 1
#if USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"

extern "C" {
void kernel_gridder(
	const int jobsize, const int bl_offset,
	const UVWType		   __restrict__ *uvw,
	const WavenumberType   __restrict__ *wavenumbers,
	const VisibilitiesType __restrict__ *visibilities,
	const SpheroidalType   __restrict__ *spheroidal,
	const ATermType		   __restrict__ *aterm,
	const BaselineType	   __restrict__ *baselines,		
	SubGridType			   __restrict__ *subgrid
	) {
	
	omp_set_nested(1);
	
    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, baselines)
    {
    #if USE_LIKWID
    likwid_markerThreadInit();
    likwid_markerStartRegion("gridder");
    #endif
    #pragma omp for
	for (int bl = 0; bl < jobsize; bl++) {
	    // Load stations for current baseline
	    int station1 = (*baselines)[bl+bl_offset].station1;
	    int station2 = (*baselines)[bl+bl_offset].station2;
	
	    for (int chunk = 0; chunk < NR_CHUNKS; chunk++) {
            // Initialize private uv grid
		    FLOAT_COMPLEX uv[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
		    memset(uv, 0, SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));
	
	        // Storage for precomputed values
	        float phase_index[SUBGRIDSIZE][SUBGRIDSIZE]  __attribute__((aligned(32)));
	        float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
	        FLOAT_COMPLEX visXX[NR_CHANNELS]             __attribute__((aligned(32)));
            FLOAT_COMPLEX visXY[NR_CHANNELS]             __attribute__((aligned(32)));
            FLOAT_COMPLEX visYX[NR_CHANNELS]             __attribute__((aligned(32)));
            FLOAT_COMPLEX visYY[NR_CHANNELS]             __attribute__((aligned(32)));
		    float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));
        	float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));
        	
	        // Compute offset for current chunk
            UVW uvw_first = (*uvw)[bl][0];
            UVW uvw_last  = (*uvw)[bl][CHUNKSIZE - 1];
		    int u = (GRIDSIZE/2) - ((uvw_first.u + uvw_last.u) / 2);
		    int v = (GRIDSIZE/2) - ((uvw_first.v + uvw_last.v) / 2);
		    int w = (GRIDSIZE/2) - ((uvw_first.w + uvw_last.w) / 2);
		    UVW _offset = make_UVW(u, v, w);
		    
		    // Iterate all timesteps in current chunk
		    int time_offset = chunk * CHUNKSIZE;
		    for (int time = time_offset; time < time_offset + CHUNKSIZE && time < NR_TIME; time++) {
			    // Load UVW coordinates
			    float u = (*uvw)[bl][time].u;
			    float v = (*uvw)[bl][time].v;
			    float w = (*uvw)[bl][time].w;
		
		        // Compute phase indices and phase offsets
                #pragma unroll
			    for (int y = 0; y < SUBGRIDSIZE; y++) {
				    for (int x = 0; x < SUBGRIDSIZE; x++) {
					    // Compute l,m,n
					    float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
					    float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
					    float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
					
					    // Compute phase index
            			phase_index[y][x] = u*l + v*m + w*n;
					
					    // Compute phase offset
					    phase_offset[y][x] = _offset.u*l + _offset.v*m + _offset.w*n;
                    }
                }
			
			    // Load visibilities
			    for (int chan = 0; chan < NR_CHANNELS; chan++) {
			        visXX[chan] = (*visibilities)[bl][time][chan][0];
				    visXY[chan] = (*visibilities)[bl][time][chan][1];
				    visYX[chan] = (*visibilities)[bl][time][chan][2];
				    visYY[chan] = (*visibilities)[bl][time][chan][3];
			    }
			
			    // Compute phasor
			    #if USE_VML
			    float phase[SUBGRIDSIZE][SUBGRIDSIZE][NR_CHANNELS] __attribute__((aligned(32)));

			    #pragma unroll_and_jam(3)
			    for (int y = 0; y < SUBGRIDSIZE; y++) {
				    for (int x = 0; x < SUBGRIDSIZE; x++) {
					    for (int chan = 0; chan < NR_CHANNELS; chan++) {
						    phase[y][x][chan] = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
					    }
				    }
			    }
			
			    vmsSinCos(SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS, (const float*) &phase[0][0][0], &phasor_imag[0][0][0], &phasor_real[0][0][0], VML_PRECISION);
			    #else
			    #pragma unroll_and_jam(3)
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
                #endif
		
		        // Update current subgrid
			    for (int y = 0; y < SUBGRIDSIZE; y++) {
				    for (int x = 0; x < SUBGRIDSIZE; x++) {
				        FLOAT_COMPLEX phasor[NR_CHANNELS] __attribute__((aligned(32)));
				        #pragma unroll
				        for (int chan = 0; chan < NR_CHANNELS; chan++) {
				            phasor[chan] = FLOAT_COMPLEX(phasor_real[y][x][chan], phasor_imag[y][x][chan]);
				        }
				
				        #pragma unroll
					    for (int chan = 0; chan < NR_CHANNELS; chan++) {
                            uv[y][x][0] += visXX[chan] * phasor[chan];
						    uv[y][x][1] += visXY[chan] * phasor[chan];
						    uv[y][x][2] += visYX[chan] * phasor[chan];
						    uv[y][x][3] += visYY[chan] * phasor[chan];
					    }
				    }
			    }
		    }
		
		    // Apply aterm and spheroidal and store result
		    for (int y = 0; y < SUBGRIDSIZE; y++) {
		        #pragma ivdep
			    for (int x = 0; x < SUBGRIDSIZE; x++) {
				    // Get a term for station1
				    FLOAT_COMPLEX aXX1 = (*aterm)[station1][0][y][x];
				    FLOAT_COMPLEX aXY1 = (*aterm)[station1][1][y][x];
				    FLOAT_COMPLEX aYX1 = (*aterm)[station1][2][y][x];
				    FLOAT_COMPLEX aYY1 = (*aterm)[station1][3][y][x];
	
				    // Get aterm for station2
				    FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][0][y][x]);
				    FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][1][y][x]);
				    FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][2][y][x]);
				    FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][3][y][x]);
	
	                // Get spheroidal
	                float s = (*spheroidal)[y][x];
	                
				    // Load uv values
                    FLOAT_COMPLEX uvXX = uv[y][x][0];
				    FLOAT_COMPLEX uvXY = uv[y][x][1];
				    FLOAT_COMPLEX uvYX = uv[y][x][2];
				    FLOAT_COMPLEX uvYY = uv[y][x][3];

				    // Apply aterm to subgrid
                    uv[y][x][0]  = (uvXX * aXX1);
                    uv[y][x][0] += (uvXY * aYX1);
                    uv[y][x][0] += (uvXX * aXX2);
                    uv[y][x][0] += (uvYX * aYX2);
                    uv[y][x][1]  = (uvXX * aXY1);
                    uv[y][x][1] += (uvXY * aYY1);
                    uv[y][x][1] += (uvXY * aXX2);
                    uv[y][x][1] += (uvYY * aYX2);
                    uv[y][x][2]  = (uvYX * aXX1);
                    uv[y][x][2] += (uvYY * aYX1);
                    uv[y][x][2] += (uvXX * aXY2); 
                    uv[y][x][2] += (uvYX * aYY2);
                    uv[y][x][3]  = (uvYX * aXY1);
                    uv[y][x][3] += (uvYY * aYY1);
                    uv[y][x][3] += (uvXY * aXY2); 
                    uv[y][x][3] += (uvYY * aYY2);
                                    
                    // Compute shifted position in subgrid
                    int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    
		            // Update uv grid
		            #if ORDER == ORDER_BL_V_U_P
		            (*subgrid)[bl][chunk][y_dst][x_dst][0] = uv[y][x][0] * s;
		            (*subgrid)[bl][chunk][y_dst][x_dst][1] = uv[y][x][1] * s;
		            (*subgrid)[bl][chunk][y_dst][x_dst][2] = uv[y][x][2] * s;
		            (*subgrid)[bl][chunk][y_dst][x_dst][3] = uv[y][x][3] * s;
		            #elif ORDER == ORDER_BL_P_V_U
		            (*subgrid)[bl][chunk][0][y_dst][x_dst] = uv[y][x][0] * s;
		            (*subgrid)[bl][chunk][1][y_dst][x_dst] = uv[y][x][1] * s;
		            (*subgrid)[bl][chunk][2][y_dst][x_dst] = uv[y][x][2] * s;
		            (*subgrid)[bl][chunk][3][y_dst][x_dst] = uv[y][x][3] * s;
                    #endif
	            }
            }
        }
	}
    #if USE_LIKWID
    likwid_markerStopRegion("gridder");
    #endif
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
