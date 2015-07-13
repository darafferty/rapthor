#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#define USE_VML 0 // Hack
#if USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"

extern "C" {
void kernel_degridder(
    const int jobsize, const int bl_offset,
	const SubGridType	 __restrict__ *subgrid,
	const UVWType		 __restrict__ *uvw,
	const WavenumberType __restrict__ *wavenumbers,
	const ATermType		 __restrict__ *aterm,
	const BaselineType	 __restrict__ *baselines,
	const SpheroidalType __restrict__ *spheroidal,
	VisibilitiesType	 __restrict__ *visibilities
	) {

  printf("Running: kernel_degridder\n");

    #pragma omp parallel shared(subgrid, uvw, wavenumbers, aterm, baselines, spheroidal)
    {
    #if USE_LIKWID
    likwid_markerThreadInit();
    likwid_markerStartRegion("degridder");
    #endif
    #pragma omp for
	for (int bl = 0; bl < jobsize; bl++) {
	    // Load stations for current baseline
	    int station1 = (*baselines)[bl+bl_offset].station1;
	    int station2 = (*baselines)[bl+bl_offset].station2;
	
	    for (int chunk = 0; chunk < NR_CHUNKS; chunk++) {
	        // Storage for precomputed values
            FLOAT_COMPLEX _subgrid[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS] __attribute__((aligned(32)));
		    float phasor_real[NR_CHANNELS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        	float phasor_imag[NR_CHANNELS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        	float phase_index[SUBGRIDSIZE][SUBGRIDSIZE]  __attribute__((aligned(32)));
	        float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
	
	        // Compute offset for current chunk
            UVW uvw_first = (*uvw)[bl][0];
            UVW uvw_last  = (*uvw)[bl][CHUNKSIZE - 1];
            int u = (GRIDSIZE/2) - ((uvw_first.u + uvw_last.u) / 2);
		    int v = (GRIDSIZE/2) - ((uvw_first.v + uvw_last.v) / 2);
		    int w = (GRIDSIZE/2) - ((uvw_first.w + uvw_last.w) / 2);
		    UVW _offset = make_UVW(u, v, w);
	
            // Apply aterm to subgrid
		    for (int y = 0; y < SUBGRIDSIZE; y++) {
			    for (int x = 0; x < SUBGRIDSIZE; x++) {
	        		// Get aterm for station1
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
		
                    // Compute shifted position in subgrid
                    int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
        	
			        // Load uv values
                    #if ORDER == ORDER_BL_P_V_U
				    FLOAT_COMPLEX uvXX = s * (*subgrid)[bl][chunk][0][y_src][x_src];
				    FLOAT_COMPLEX uvXY = s * (*subgrid)[bl][chunk][1][y_src][x_src];
				    FLOAT_COMPLEX uvYX = s * (*subgrid)[bl][chunk][2][y_src][x_src];
				    FLOAT_COMPLEX uvYY = s * (*subgrid)[bl][chunk][3][y_src][x_src];
			        #elif ORDER == ORDER_BL_V_U_P
				    FLOAT_COMPLEX uvXX = s * (*subgrid)[bl][chunk][y_src][x_src][0];
				    FLOAT_COMPLEX uvXY = s * (*subgrid)[bl][chunk][y_src][x_src][1];
				    FLOAT_COMPLEX uvYX = s * (*subgrid)[bl][chunk][y_src][x_src][2];
				    FLOAT_COMPLEX uvYY = s * (*subgrid)[bl][chunk][y_src][x_src][3];
				    #endif
			
                    // Apply aterm to subgrid
				    _subgrid[y][x][0]  = uvXX * aXX1;
                    _subgrid[y][x][0] += uvXY * aYX1;
                    _subgrid[y][x][0] += uvXX * aXX2;
                    _subgrid[y][x][0] += uvYX * aYX2;
				    _subgrid[y][x][1]  = uvXX * aXY1;
                    _subgrid[y][x][1] += uvXY * aYY1;
                    _subgrid[y][x][1] += uvXY * aXX2;
                    _subgrid[y][x][1] += uvYY * aYX2;
				    _subgrid[y][x][2]  = uvYX * aXX1;
                    _subgrid[y][x][2] += uvYY * aYX1;
                    _subgrid[y][x][2] += uvXX * aXY2;
                    _subgrid[y][x][2] += uvYX * aYY2;
				    _subgrid[y][x][3]  = uvYX * aXY1;
                    _subgrid[y][x][3] += uvYY * aYY1;
                    _subgrid[y][x][3] += uvXY * aXY2;
                    _subgrid[y][x][3] += uvYY * aYY2;
			    }
		    }
	
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
					    phase_offset[y][x] = _offset.u*l +
									         _offset.v*m +
										     _offset.w*n;
                    }
                }
			
			    // Compute phasor
		        #if USE_VML
		        float phase[NR_CHANNELS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

		        #pragma unroll_and_jam(3)
		        for (int chan = 0; chan < NR_CHANNELS; chan++) {
		            for (int y = 0; y < SUBGRIDSIZE; y++) {
			            for (int x = 0; x < SUBGRIDSIZE; x++) {
					        phase[chan][y][x] = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
				        }
			        }
		        }
		
		        vmsSinCos(SUBGRIDSIZE * SUBGRIDSIZE * NR_CHANNELS, (const float*) &phase[0][0][0], &phasor_imag[0][0][0], &phasor_real[0][0][0], VML_PRECISION);
		        #else
		        #pragma unroll_and_jam(3)
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
                #endif
            
		        FLOAT_COMPLEX sum[NR_POLARIZATIONS] __attribute__((aligned(32)));
        
			    for (int chan = 0; chan < NR_CHANNELS; chan++) {
		            memset(sum, 0, NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));
			
		            for (int y = 0; y < SUBGRIDSIZE; y++) {
				        for (int x = 0; x < SUBGRIDSIZE; x++) {
				            FLOAT_COMPLEX phasor = FLOAT_COMPLEX(phasor_real[chan][y][x], phasor_imag[chan][y][x]);

						    // Update all polarizations
						    sum[0] += _subgrid[y][x][0] * phasor;
						    sum[1] += _subgrid[y][x][1] * phasor;
						    sum[2] += _subgrid[y][x][2] * phasor;
						    sum[3] += _subgrid[y][x][3] * phasor;
					    }
				    }
				
				    // Set visibilities
				    (*visibilities)[bl][time][chan][0] = sum[0];
				    (*visibilities)[bl][time][chan][1] = sum[1];
				    (*visibilities)[bl][time][chan][2] = sum[2];
				    (*visibilities)[bl][time][chan][3] = sum[3];
			    }
		    }
	    }
	}
    #if USE_LIKWID
    likwid_markerStopRegion("degridder");
    #endif
    }
}

uint64_t kernel_degridder_flops(int jobsize) {
    return
    // ATerm
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 32 +
    // Shift
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 6 +
    // Spheroidal
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2 +
    // Degrid
    1ULL * jobsize * NR_TIME * NR_CHANNELS * SUBGRIDSIZE * SUBGRIDSIZE * (
        // Phasor
        2 * 22 +
        // UV
        NR_POLARIZATIONS * 8);
}

uint64_t kernel_degridder_bytes(int jobsize) {
    return
    // ATerm
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * 2 * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) +
    // Spheroidal
    1ULL * jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(float) +
    // Degrid
    1ULL * jobsize * NR_TIME * NR_CHANNELS * (
        // Offset
        SUBGRIDSIZE * SUBGRIDSIZE * 3 * sizeof(float) +
        // UV
        SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX) +
        // Visibilities            
        NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));
}
}
