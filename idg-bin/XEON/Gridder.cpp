#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#if USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include <idg/Common/Types.h>

extern "C" {

/*
	Kernel
*/
void kernel_gridder(
	const int jobsize,
	const UVWType			__restrict__ *uvw,
	const OffsetType		__restrict__ *offset,
	const WavenumberType	__restrict__ *wavenumbers,
	const VisibilitiesType	__restrict__ *visibilities,
	const SpheroidalType	__restrict__ *spheroidal,
	const ATermType			__restrict__ *aterm,
	const BaselineType		__restrict__ *baselines,		
	UVGridType				__restrict__ *uvgrid
	) {
    #pragma omp parallel for shared(uvw, offset, wavenumbers, visibilities, spheroidal, aterm, baselines)
	for (int bl = 0; bl < jobsize; bl++) {
        // Initialize private uv grid
		std::complex<float> uv[BLOCKSIZE][BLOCKSIZE][NR_POLARIZATIONS];
		memset(uv, 0, BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(std::complex<float>));
	
	    // Offset for current baseline
		UVW _offset = (*offset)[bl];
	
	    // Storage for precomputed values
	    float phase_index[BLOCKSIZE][BLOCKSIZE]  __attribute__((aligned(32)));
	    float phase_offset[BLOCKSIZE][BLOCKSIZE] __attribute__((aligned(32)));
	    std::complex<float> visXX[NR_CHANNELS]         __attribute__((aligned(32)));
        std::complex<float> visXY[NR_CHANNELS]         __attribute__((aligned(32)));
        std::complex<float> visYX[NR_CHANNELS]         __attribute__((aligned(32)));
        std::complex<float> visYY[NR_CHANNELS]         __attribute__((aligned(32)));
		float phasor_real[BLOCKSIZE][BLOCKSIZE][NR_CHANNELS] __attribute__((aligned(32)));
    	float phasor_imag[BLOCKSIZE][BLOCKSIZE][NR_CHANNELS] __attribute__((aligned(32)));
	
		// Apply visibilities to uvgrid
		for (int time = 0; time < NR_TIME; time++) {
			// Load UVW coordinates
			float u = (*uvw)[bl][time].u;
			float v = (*uvw)[bl][time].v;
			float w = (*uvw)[bl][time].w;
		
		    // Compute phase indices and phase offsets
            #pragma unroll
			for (int y = 0; y < BLOCKSIZE; y++) {
				for (int x = 0; x < BLOCKSIZE; x++) {
					// Compute l,m,n
					float l = -(x-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
					float m =  (y-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
					float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
					
					// Compute phase index
        			phase_index[y][x] = u*l + v*m + w*n;
					
					// Compute phase offset
					phase_offset[y][x] = _offset.u*l +
									     _offset.v*m +
										 _offset.w*n;
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
			float phase[BLOCKSIZE][BLOCKSIZE][NR_CHANNELS] __attribute__((aligned(32)));

			#pragma unroll_and_jam(3)
			for (int y = 0; y < BLOCKSIZE; y++) {
				for (int x = 0; x < BLOCKSIZE; x++) {
					for (int chan = 0; chan < NR_CHANNELS; chan++) {
						phase[y][x][chan] = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
					}
				}
			}
			
			vmsSinCos(BLOCKSIZE * BLOCKSIZE * NR_CHANNELS, (const float*) &phase[0][0][0], &phasor_imag[0][0][0], &phasor_real[0][0][0], VML_LA);
			#else
			#pragma unroll_and_jam(3)
			for (int y = 0; y < BLOCKSIZE; y++) {
				for (int x = 0; x < BLOCKSIZE; x++) {
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
		
		    // Update current uvgrid
			for (int y = 0; y < BLOCKSIZE; y++) {
				for (int x = 0; x < BLOCKSIZE; x++) {
				    std::complex<float> phasor[NR_CHANNELS] __attribute__((aligned(32)));
				    #pragma unroll
				    for (int chan = 0; chan < NR_CHANNELS; chan++) {
				        phasor[chan] = std::complex<float>(phasor_real[y][x][chan], phasor_imag[y][x][chan]);
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
		
		// Stations for current baseline
		int station1 = (*baselines)[bl].station1;
		int station2 = (*baselines)[bl].station2;
		
		// Apply aterm and spheroidal and store result
		#pragma unroll_and_jam(2)
		for (int y = 0; y < BLOCKSIZE; y++) {
		    #pragma ivdep
			for (int x = 0; x < BLOCKSIZE; x++) {
				// Get a term for station1
				std::complex<float> aXX1 = (*aterm)[station1][0][y][x];
				std::complex<float> aXY1 = (*aterm)[station1][1][y][x];
				std::complex<float> aYX1 = (*aterm)[station1][2][y][x];
				std::complex<float> aYY1 = (*aterm)[station1][3][y][x];
	
				// Get aterm for station2
				std::complex<float> aXX2 = conj((*aterm)[station2][0][y][x]);
				std::complex<float> aXY2 = conj((*aterm)[station2][1][y][x]);
				std::complex<float> aYX2 = conj((*aterm)[station2][2][y][x]);
				std::complex<float> aYY2 = conj((*aterm)[station2][3][y][x]);
	
	            // Get spheroidal
	            float s = (*spheroidal)[y][x];
	            
				// Load uv values
                std::complex<float> uvXX = uv[y][x][0];
				std::complex<float> uvXY = uv[y][x][1];
				std::complex<float> uvYX = uv[y][x][2];
				std::complex<float> uvYY = uv[y][x][3];

				// Apply aterm to uvgrid
                uv[y][x][0] += (uvXX * aXX1);
                uv[y][x][0] += (uvXY * aYX1);
                uv[y][x][0] += (uvXX * aXX2);
                uv[y][x][0] += (uvYX * aYX2);
                uv[y][x][1] += (uvXX * aXY1);
                uv[y][x][1] += (uvXY * aYY1);
                uv[y][x][1] += (uvXY * aXX2);
                uv[y][x][1] += (uvYY * aYX2);
                uv[y][x][2] += (uvYX * aXX1);
                uv[y][x][2] += (uvYY * aYX1);
                uv[y][x][2] += (uvXX * aXY2); 
                uv[y][x][2] += (uvYX * aYY2);
                uv[y][x][3] += (uvYX * aXY1);
                uv[y][x][3] += (uvYY * aYY1);
                uv[y][x][3] += (uvXY * aXY2); 
                uv[y][x][3] += (uvYY * aYY2);
                                
		        // Update uv grid
		        #if ORDER == ORDER_BL_V_U_P
		        (*uvgrid)[bl][y][x][0] = uv[y][x][0] * s;
		        (*uvgrid)[bl][y][x][1] = uv[y][x][1] * s;
		        (*uvgrid)[bl][y][x][2] = uv[y][x][2] * s;
		        (*uvgrid)[bl][y][x][3] = uv[y][x][3] * s;
		        #elif ORDER == ORDER_BL_P_V_U
		        (*uvgrid)[bl][0][y][x] = uv[y][x][0] * s;
		        (*uvgrid)[bl][1][y][x] = uv[y][x][1] * s;
		        (*uvgrid)[bl][2][y][x] = uv[y][x][2] * s;
		        (*uvgrid)[bl][3][y][x] = uv[y][x][3] * s;
                #endif
	        }
        }
	}
}

#include "idg/Common/Parameters.h"

}

