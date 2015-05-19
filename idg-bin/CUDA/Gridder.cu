#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_gridder(
	const int bl_offset,
	const UVWType			__restrict__ uvw,
	const OffsetType		__restrict__ offset,
	const WavenumberType	__restrict__ wavenumbers,
	const VisibilitiesType	__restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const BaselineType		__restrict__ baselines,		
	UVGridType				__restrict__ uvgrid
	) {
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidx + tidy * blockDim.x;
	int bl = blockIdx.x + blockIdx.y * gridDim.x;
	
    // Shared data
	__shared__ UVW _uvw[NR_TIME];
	__shared__ UVW _offset;
	__shared__ float _wavenumbers[NR_CHANNELS];
	__shared__ float2 _visibilities[NR_CHANNELS][NR_POLARIZATIONS];
	
    // Load UVW	
	if (tid < NR_TIME) {
	    _uvw[tid] = uvw[bl][tid];	    
	}
	tid -= NR_TIME;
	
	// Load wavenumbers
	if (tid >= 0 && tid < NR_CHANNELS) {
	    _wavenumbers[tid] = wavenumbers[tid];
	}
	tid -= NR_CHANNELS;
	
	// Load offset
	if (tid == 0) {
	    _offset = offset[bl+bl_offset];
	}

    // Restore tid	
	tid = tidx + tidy * blockDim.x;
	
	__syncthreads();
	
    for (int y = tidy; y < BLOCKSIZE; y += blockDim.y) {
	    for (int x = tidx; x < BLOCKSIZE; x += blockDim.x) {
	        // Private uvgrid points
	        float2 uvXX = {0, 0};
	        float2 uvXY = {0, 0};
	        float2 uvYX = {0, 0};
	        float2 uvYY = {0, 0};
	    
	        // Compute l,m,n
	        float l = -(x-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
	        float m =  (y-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

	
            // Apply visibilities for all timesteps to uvgrid point
	        for (int time = 0; time < NR_TIME; time++) {
	        	// Load visibilities
            	for (int i = tid; i < NR_CHANNELS * NR_POLARIZATIONS; i += blockDim.x * blockDim.y) {
            	    _visibilities[0][i] = visibilities[bl][time][0][i];
            	}
	        
	             // Load UVW coordinates
		        float u = _uvw[time].u;
		        float v = _uvw[time].v;
		        float w = _uvw[time].w;
		
		        // Compute phase index
            	float ulvmwn = u*l + v*m + w*n;

		        // Compute phase offset
		        float phase_offset = _offset.u*l +
						             _offset.v*m +
						             _offset.w*n;
						             
		        // Compute phasor
		        float phasor_real[NR_CHANNELS];
		        float phasor_imag[NR_CHANNELS];
		        for (int chan = 0; chan < NR_CHANNELS; chan++) {
		            float phase = (ulvmwn * _wavenumbers[chan]) - phase_offset;
                    float2 phasor = make_float2(0, 0);
                    phasor_real[chan] = cos(phase);
                    phasor_imag[chan] = sin(phase);
                }

		        // Sum updates for all channels
		        for (int chan = 0; chan < NR_CHANNELS; chan++) {
			        // Load visibilities from shared memory
			        float2 visXX = _visibilities[chan][0];
			        float2 visXY = _visibilities[chan][1];
			        float2 visYX = _visibilities[chan][2];
			        float2 visYY = _visibilities[chan][3];
			            	
			        // Load phasor
                    float2 phasor = make_float2(phasor_real[chan], phasor_imag[chan]);
			
			        // Multiply visibility by phasor
			        uvXX += phasor * visXX;
			        uvXY += phasor * visXY;
			        uvYX += phasor * visYX;
			        uvYY += phasor * visYY;
		        }
	        }
	
	        // Get spheroidal
	        float s = spheroidal[y][x];
	
	        // Stations for current baseline
            int station1 = baselines[bl+bl_offset].station1;
	        int station2 = baselines[bl+bl_offset].station2;
	
            // Get a term for station1
	        float2 aXX1 = aterm[station1][0][y][x];
	        float2 aXY1 = aterm[station1][1][y][x];
	        float2 aYX1 = aterm[station1][2][y][x];
	        float2 aYY1 = aterm[station1][3][y][x];

	        // Get aterm for station2
	        float2 aXX2 = cuConjf(aterm[station2][0][y][x]);
	        float2 aXY2 = cuConjf(aterm[station2][1][y][x]);
	        float2 aYX2 = cuConjf(aterm[station2][2][y][x]);
	        float2 aYY2 = cuConjf(aterm[station2][3][y][x]);
	
	        // Apply aterm
	        float2 _uvXX = uvXX;
	        float2 _uvXY = uvXY;
	        float2 _uvYX = uvYX;
	        float2 _uvYY = uvYY;
	        uvXX += _uvXX * aXX1;
	        uvXX += _uvXY * aYX1; 
	        uvXX += _uvXX * aXX2;
	        uvXX += _uvXY * aYX2;
	        uvXY += _uvXX * aXY1;
	        uvXY += _uvXY * aYY1;
	        uvXY += _uvXX * aXY2;
	        uvXY += _uvXY * aYY2;
	        uvYX += _uvYX * aXX1;
	        uvYX += _uvYY * aYX1;
	        uvYX += _uvYX * aXX2;
	        uvYX += _uvYY * aYX2;
	        uvYY += _uvYY * aXY1;
	        uvYY += _uvYY * aYY1;
	        uvYY += _uvYY * aXY2;
	        uvYY += _uvYY * aYY2;
	
	        // Apply spheroidal and update uv grid
            #if ORDER == ORDER_BL_P_V_U
	        uvgrid[bl][0][y][x] = uvXX * s;
	        uvgrid[bl][1][y][x] = uvXY * s;
	        uvgrid[bl][2][y][x] = uvYX * s;
	        uvgrid[bl][3][y][x] = uvYY * s;
	        #elif ORDER_BL_V_U_P
	        uvgrid[bl][y][x][0] = uvXX * s;
	        uvgrid[bl][y][x][1] = uvXY * s;
	        uvgrid[bl][y][x][2] = uvYX * s;
	        uvgrid[bl][y][x][3] = uvYY * s;
	        #endif
        }
    }
}
}
