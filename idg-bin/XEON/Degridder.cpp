#include <math.h>
#include <stdio.h>
#include <immintrin.h>
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
void kernel_degridder(
    int jobsize,
	const UVGridType	    __restrict__ *uvgrid,
	const UVWType			__restrict__ *uvw,
	const OffsetType		__restrict__ *offset,
	const WavenumberType	__restrict__ *wavenumbers,
	const ATermType			__restrict__ *aterm,
	const BaselineType		__restrict__ *baselines,
	const SpheroidalType    __restrict__ *spheroidal,
	VisibilitiesType	    __restrict__ *visibilities
	) {
    #pragma omp parallel for shared(uvgrid, uvw, offset, wavenumbers, aterm, baselines, spheroidal)
	for (int bl = 0; bl < jobsize; bl++) {
    	 // Offset for current baseline
		UVW _offset = (*offset)[bl];
	
	    // Storage for precomputed values
        std::complex<float> uv[BLOCKSIZE][BLOCKSIZE][NR_POLARIZATIONS] __attribute__((aligned(32)));
		float phasor_real[NR_CHANNELS][BLOCKSIZE][BLOCKSIZE] __attribute__((aligned(32)));
    	float phasor_imag[NR_CHANNELS][BLOCKSIZE][BLOCKSIZE] __attribute__((aligned(32)));
    	float phase_index[BLOCKSIZE][BLOCKSIZE]  __attribute__((aligned(32)));
	    float phase_offset[BLOCKSIZE][BLOCKSIZE] __attribute__((aligned(32)));
	
		// Stations for current baseline
		int station1 = (*baselines)[bl].station1;
		int station2 = (*baselines)[bl].station2;
		
        // Apply aterm to uvgrid
		for (int y = 0; y < BLOCKSIZE; y++) {
			for (int x = 0; x < BLOCKSIZE; x++) {
	    		// Get aterm for station1
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
                #if ORDER == ORDER_BL_P_V_U
				std::complex<float> uvXX = s*(*uvgrid)[bl][0][y][x];
				std::complex<float> uvXY = s*(*uvgrid)[bl][1][y][x];
				std::complex<float> uvYX = s*(*uvgrid)[bl][2][y][x];
				std::complex<float> uvYY = s*(*uvgrid)[bl][3][y][x];
			    #elif ORDER == ORDER_BL_V_U_P
				std::complex<float> uvXX = s*(*uvgrid)[bl][y][x][0];
				std::complex<float> uvXY = s*(*uvgrid)[bl][y][x][1];
				std::complex<float> uvYX = s*(*uvgrid)[bl][y][x][2];
				std::complex<float> uvYY = s*(*uvgrid)[bl][y][x][3];
				#endif
			
                // Apply aterm to uvgrid
				uv[y][x][0] = uvXX * aXX1;
                uv[y][x][0] += uvXY * aYX1;
                uv[y][x][0] += uvXX * aXX2;
                uv[y][x][0] += uvYX * aYX2;
				uv[y][x][1] = uvXX * aXY1;
                uv[y][x][1] += uvXY * aYY1;
                uv[y][x][1] += uvXY * aXX2;
                uv[y][x][1] += uvYY * aYX2;
				uv[y][x][2] = uvYX * aXX1;
                uv[y][x][2] += uvYY * aYX1;
                uv[y][x][2] += uvXX * aXY2;
                uv[y][x][2] += uvYX * aYY2;
				uv[y][x][3] = uvYX * aXY1;
                uv[y][x][3] += uvYY * aYY1;
                uv[y][x][3] += uvXY * aXY2;
                uv[y][x][3] += uvYY * aYY2;
                
			}
		}
	
		// Exctract visibilities from uvgrid
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
					float l = (x-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
					float m = -(y-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
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
		    float phase[NR_CHANNELS][BLOCKSIZE][BLOCKSIZE] __attribute__((aligned(32)));

		    #pragma unroll_and_jam(3)
		    for (int chan = 0; chan < NR_CHANNELS; chan++) {
		        for (int y = 0; y < BLOCKSIZE; y++) {
			        for (int x = 0; x < BLOCKSIZE; x++) {
					    phase[chan][y][x] = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
				    }
			    }
		    }
		
		    vmsSinCos(BLOCKSIZE * BLOCKSIZE * NR_CHANNELS, (const float*) &phase[0][0][0], &phasor_imag[0][0][0], &phasor_real[0][0][0], VML_PRECISION);
		    #else
		    #pragma unroll_and_jam(3)
		    for (int chan = 0; chan < NR_CHANNELS; chan++) {
		        for (int y = 0; y < BLOCKSIZE; y++) {
			        for (int x = 0; x < BLOCKSIZE; x++) {
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
        
		    std::complex<float> sum[NR_POLARIZATIONS] __attribute__((aligned(32)));
    
			for (int chan = 0; chan < NR_CHANNELS; chan++) {
		        memset(sum, 0, NR_POLARIZATIONS * sizeof(std::complex<float>));
			
		        for (int y = 0; y < BLOCKSIZE; y++) {
				    for (int x = 0; x < BLOCKSIZE; x++) {
				        std::complex<float> phasor(phasor_real[chan][y][x], phasor_imag[chan][y][x]);

						// Update all polarizations
						sum[0] += uv[y][x][0] * phasor;
						sum[1] += uv[y][x][1] * phasor;
						sum[2] += uv[y][x][2] * phasor;
						sum[3] += uv[y][x][3] * phasor;
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

#include <idg/Common/Parameters.h>

}
