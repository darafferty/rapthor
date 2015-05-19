#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {


/*
	Kernel
*/
__global__ void kernel_degridder(
	const int bl_offset,
	const UVGridType	    __restrict__ uvgrid,
	const UVWType			__restrict__ uvw,
	const OffsetType		__restrict__ offset,
	const WavenumberType	__restrict__ wavenumbers,
	const ATermType			__restrict__ aterm,
	const BaselineType		__restrict__ baselines,
	const SpheroidalType    __restrict__ spheroidal,
	VisibilitiesType	    __restrict__ visibilities
	) {
	int bl = blockIdx.x + blockIdx.y * gridDim.x;
    // Load offset
    UVW _offset = offset[bl];
	
	// Stations for current baseline
	int station1 = baselines[bl+bl_offset].station1;
	int station2 = baselines[bl+bl_offset].station2;
    
	 // Shared data
    __shared__ float2 _uv[BLOCKSIZE][NR_POLARIZATIONS];
    __shared__ UVW _lmn[BLOCKSIZE];
    
    for (int i = threadIdx.x; i < NR_TIME * NR_CHANNELS; i += blockDim.x) {
    	int time = i / NR_CHANNELS;
    	int chan = i % NR_CHANNELS;
    
    	 // Private data points
	    float2 dataXX = {0, 0};
	    float2 dataXY = {0, 0};
	    float2 dataYX = {0, 0};
	    float2 dataYY = {0, 0};
    
        // Load wavenumber
        float wavenumber = wavenumbers[chan];
    
    	// Load UVW coordinate
	    float u = uvw[bl][time].u;
        float v = uvw[bl][time].v;
        float w = uvw[bl][time].w;
        
        for (int y = 0; y < BLOCKSIZE; y++) {

            for (int x = threadIdx.x; x < BLOCKSIZE; x += blockDim.x) {
            	// Compute l,m,n
                float l = -(x-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
                float m =  (y-(BLOCKSIZE/2)) * IMAGESIZE/BLOCKSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                
                // Store l,m,n
                _lmn[x].u = l;
                _lmn[x].v = m;
                _lmn[x].w = n;
            }

            for (int x = threadIdx.x; x < BLOCKSIZE; x += blockDim.x) {
                // Load uv grid points
                #if ORDER == ORDER_BL_P_V_U
                float2 uvXX = uvgrid[bl][0][y][x];
                float2 uvXY = uvgrid[bl][1][y][x];
                float2 uvYX = uvgrid[bl][2][y][x];
                float2 uvYY = uvgrid[bl][3][y][x];
                #elif ORDER == ORDER_BL_V_U_P
                float2 uvXX = uvgrid[bl][y][x][0];
                float2 uvXY = uvgrid[bl][y][x][1];
                float2 uvYX = uvgrid[bl][y][x][2];
                float2 uvYY = uvgrid[bl][y][x][3];
                #endif
                
                // Get spheroidal
                float s = spheroidal[y][x];
                
               	// Get aterm for station1
                float2 aXX1 = aterm[station1][0][y][x];
                float2 aXY1 = aterm[station1][1][y][x];
                float2 aYX1 = aterm[station1][2][y][x];
                float2 aYY1 = aterm[station1][3][y][x];

                // Get aterm for station2
                float2 aXX2 = cuConjf(aterm[station2][0][y][x]);
                float2 aXY2 = cuConjf(aterm[station2][1][y][x]);
                float2 aYX2 = cuConjf(aterm[station2][2][y][x]);
                float2 aYY2 = cuConjf(aterm[station2][3][y][x]);
                
                // Initialize corected uv values
                float2 _uvXX = uvXX * s;
                float2 _uvXY = uvXY * s;
                float2 _uvYX = uvYX * s;
                float2 _uvYY = uvYY * s;
                
                // Apply aterm
            	_uvXX += uvXX * aXX1;
                _uvXX += uvXY * aYX1; 
                _uvXX += uvXX * aXX2;
                _uvXX += uvXY * aYX2;
                _uvXY += uvXX * aXY1;
                _uvXY += uvXY * aYY1;
                _uvXY += uvXX * aXY2;
                _uvXY += uvXY * aYY2;
                _uvYX += uvYX * aXX1;
                _uvYX += uvYY * aYX1;
                _uvYX += uvYX * aXX2;
                _uvYX += uvYY * aYX2;
                _uvYY += uvYY * aXY1;
                _uvYY += uvYY * aYY1;
                _uvYY += uvYY * aXY2;
                _uvYY += uvYY * aYY2;
                
                // Store uv values
                _uv[x][0] = _uvXX;
                _uv[x][1] = _uvXY;
                _uv[x][2] = _uvYX;
                _uv[x][3] = _uvYY;
            }
            
            __syncthreads();
            
            for (int x = 0; x < BLOCKSIZE; x++) {
                // Get l, m, n
                float l = _lmn[x].u;
                float m = _lmn[x].v;
                float n = _lmn[x].w;

                // Compute phase index
            	float ulvmwn = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = _offset.u*l + _offset.v*m + _offset.w*n;
                
                // Compute phasor
		        float2 phasor = make_float2(0, 0);
		        float phase = (ulvmwn * wavenumber) - phase_offset;
		        phasor.x = cos(phase);
		        phasor.y = sin(phase);
		
                // Update data points
                dataXX += phasor * _uv[x][0];
                dataXY += phasor * _uv[x][1];
                dataYX += phasor * _uv[x][2];
                dataYY += phasor * _uv[x][3];
            }
        }
	
	    // Set visibilities
        visibilities[bl][time][chan][0] = dataXX;
        visibilities[bl][time][chan][1] = dataXY;
        visibilities[bl][time][chan][2] = dataYX;
        visibilities[bl][time][chan][3] = dataYY;
    }
}
}
