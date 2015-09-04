#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_gridder(
	const int jobsize,
    const float w_offset,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	const VisibilitiesType	__restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,		
	SubGridType				__restrict__ subgrid
	) {
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidx + tidy * blockDim.x;
	int s = blockIdx.x;

    // Shared data
	__shared__ UVW _uvw[NR_TIMESTEPS];
	__shared__ float _wavenumbers[NR_CHANNELS];
	__shared__ float2 _visibilities[NR_CHANNELS][NR_POLARIZATIONS];
	
    // Load wavenumbers
    if (tid < NR_CHANNELS) {
        _wavenumbers[tid] = wavenumbers[tid];
    }
    __syncthreads();
	
    // Iterate all subgrids
	for (; s < jobsize; s += gridDim.x * gridDim.y) {
        // Load UVW	
        for (int time = tid; time < NR_TIMESTEPS; time += blockDim.x * blockDim.y) {
	        _uvw[time] = uvw[s][time];	    
	    }
	    __syncthreads();

	    // Load metadata
        const Metadata m = metadata[s];
        int time_nr = m.time_nr;
        int station1 = m.baseline.station1;
        int station2 = m.baseline.station2;
        int x_coordinate = m.coordinate.x;
        int y_coordinate = m.coordinate.y;

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2) / IMAGESIZE;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2) / IMAGESIZE;
	    
        // Iterate all pixels in subgrid
        for (int y = tidy; y < SUBGRIDSIZE; y += blockDim.y) {
	        for (int x = tidx; x < SUBGRIDSIZE; x += blockDim.x) {
	            // Private subgrid points
	            float2 uvXX = {0, 0};
	            float2 uvXY = {0, 0};
	            float2 uvYX = {0, 0};
	            float2 uvYY = {0, 0};
	        
	            // Compute l,m,n
	            float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
	            float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                // Iterate all timesteps
	            for (int time = 0; time < NR_TIMESTEPS; time++) {
            	    // Load visibilities for all channels and polarizations
                	for (int i = tid; i < NR_CHANNELS * NR_POLARIZATIONS; i += blockDim.x * blockDim.y) {
                	    _visibilities[0][i] = visibilities[s][time][0][i];
                	}
	            
	                 // Load UVW coordinates
		            float u = _uvw[time].u;
		            float v = _uvw[time].v;
		            float w = _uvw[time].w;
		
		            // Compute phase index
                	float ulvmwn = u*l + v*m + w*n;

		            // Compute phase offset
		            float phase_offset = u_offset*l + v_offset*m + w_offset*n;
						                 
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
	
                // Get a term for station1
	            const float2 aXX1 = aterm[station1][time_nr][0][y][x];
	            const float2 aXY1 = aterm[station1][time_nr][1][y][x];
	            const float2 aYX1 = aterm[station1][time_nr][2][y][x];
	            const float2 aYY1 = aterm[station1][time_nr][3][y][x];

	            // Get aterm for station2
	            const float2 aXX2 = cuConjf(aterm[station2][time_nr][0][y][x]);
	            const float2 aXY2 = cuConjf(aterm[station2][time_nr][1][y][x]);
	            const float2 aYX2 = cuConjf(aterm[station2][time_nr][2][y][x]);
	            const float2 aYY2 = cuConjf(aterm[station2][time_nr][3][y][x]);
	
	            // Apply aterm
	            const float2 _uvXX = uvXX;
	            const float2 _uvXY = uvXY;
	            const float2 _uvYX = uvYX;
	            const float2 _uvYY = uvYY;
	            uvXX  = _uvXX * aXX1;
	            uvXX += _uvXY * aYX1; 
	            uvXX += _uvXX * aXX2;
	            uvXX += _uvXY * aYX2;
	            uvXY  = _uvXX * aXY1;
	            uvXY += _uvXY * aYY1;
	            uvXY += _uvXX * aXY2;
	            uvXY += _uvXY * aYY2;
	            uvYX  = _uvYX * aXX1;
	            uvYX += _uvYY * aYX1;
	            uvYX += _uvYX * aXX2;
	            uvYX += _uvYY * aYX2;
	            uvYY  = _uvYY * aXY1;
	            uvYY += _uvYY * aYY1;
	            uvYY += _uvYY * aXY2;
	            uvYY += _uvYY * aYY2;

	            // Load spheroidal
	            float sph = spheroidal[y][x];
	
                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
	
	            // Set subgrid value
	            subgrid[s][0][y_dst][x_dst] = uvXX * sph;
	            subgrid[s][1][y_dst][x_dst] = uvXY * sph;
	            subgrid[s][2][y_dst][x_dst] = uvYX * sph;
	            subgrid[s][3][y_dst][x_dst] = uvYY * sph;
            }
        }
    }
}
}
