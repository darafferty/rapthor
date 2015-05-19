#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_degridder(
	const int bl_offset,
	const SubGridType	    __restrict__ subgrid,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	const ATermType			__restrict__ aterm,
	const BaselineType		__restrict__ baselines,
	const SpheroidalType    __restrict__ spheroidal,
	VisibilitiesType	    __restrict__ visibilities
	) {
    int tidx = threadIdx.x;
    int tid = tidx;
    int bl = blockIdx.x + blockIdx.y * gridDim.x;
	 
     // Shared data
    __shared__ UVW _uvw[CHUNKSIZE];
    __shared__ float _wavenumbers[NR_CHANNELS];
    __shared__ float2 _uv[SUBGRIDSIZE][NR_POLARIZATIONS];
    __shared__ float _ulvmwn[SUBGRIDSIZE];
    __shared__ float _phase_offset[SUBGRIDSIZE];
    
    // Load wavenumbers
    if (tid < NR_CHANNELS) {
        _wavenumbers[tid] = wavenumbers[tid];
    }
   
	// Stations for current baseline
	int station1 = baselines[bl+bl_offset].station1;
	int station2 = baselines[bl+bl_offset].station2;
   
    for (int chunk = 0; chunk < NR_CHUNKS; chunk++) {
        // Load UVW	
	    if (tid < CHUNKSIZE) {
	        _uvw[tid] = uvw[bl][tid];    
	    }
	    __syncthreads();
        
        // Compute offset for current chunk
	    UVW uvw_first = _uvw[0];
        UVW uvw_last = _uvw[CHUNKSIZE-1];
	    int u = (GRIDSIZE/2) - ((uvw_first.u + uvw_last.u) / 2);
	    int v = (GRIDSIZE/2) - ((uvw_first.v + uvw_last.v) / 2);
	    int w = (GRIDSIZE/2) - ((uvw_first.w + uvw_last.w) / 2);
	    UVW _offset = {u, v, w};
	  
        // Determe work distribution 
        int nr_compute_threads = SUBGRIDSIZE* 2;
        int nr_load_threads = 0;
        while (nr_load_threads <= 0) {
            nr_compute_threads /= 2;
            nr_load_threads = blockDim.x - nr_compute_threads;
        }

        for (int i = threadIdx.x; i < CHUNKSIZE * NR_CHANNELS; i += blockDim.x) {
            // Determine time and channel to be computed
            int time = (i / NR_CHANNELS);
            int chan = i % NR_CHANNELS;
        
             // Private data points
            float2 dataXX = {0, 0};
            float2 dataXY = {0, 0};
            float2 dataYX = {0, 0};
            float2 dataYY = {0, 0};
        
            // Process one row of subgrid at a time
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                if (threadIdx.x < nr_compute_threads) {
                    // A number of threads start computing phasor values
                    tid = threadIdx.x; 

                    // Precompute l,m,n for one row
                    for (int x = tid; x < SUBGRIDSIZE; x += nr_compute_threads) {
                        // Compute l,m,n
                        float l =  (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                        float m = -(y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                        float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                        
                        // Load UVW coordinate
                        float u = _uvw[time].u;
                        float v = _uvw[time].v;
                        float w = _uvw[time].w;
                
                        // Compute phase index
                        _ulvmwn[x] = u*l + v*m + w*n;

                        // Compute phase offset
                        _phase_offset[x] = _offset.u*l +
                                           _offset.v*m +
                                           _offset.w*n;
                    }
                } else {
                    // Remaining threads start preparing uvgrid values
                    tid = blockDim.x - nr_compute_threads;

                    // Preload and correct subgrid values for one row
                    for (int x = tid; x < SUBGRIDSIZE; x += nr_load_threads) {
                        // Get spheroidal
                        float s = spheroidal[y][x];
                        
                        // Load uv grid points
                        #if ORDER == ORDER_BL_P_V_U
                        float2 uvXX = s * subgrid[bl][chunk][0][y][x];
                        float2 uvXY = s * subgrid[bl][chunk][1][y][x];
                        float2 uvYX = s * subgrid[bl][chunk][2][y][x];
                        float2 uvYY = s * subgrid[bl][chunk][3][y][x];
                        #elif ORDER == ORDER_BL_V_U_P
                        float2 uvXX = s * subgrid[bl][chunk][y][x][0];
                        float2 uvXY = s * subgrid[bl][chunk][y][x][1];
                        float2 uvYX = s * subgrid[bl][chunk][y][x][2];
                        float2 uvYY = s * subgrid[bl][chunk][y][x][3];
                        #endif
                        
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
                        float2 _uvXX = {0, 0};
                        float2 _uvXY = {0, 0};
                        float2 _uvYX = {0, 0};
                        float2 _uvYY = {0, 0};
                        
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
                }
                
                __syncthreads();

                // Every thread iterates all pixels in current row 
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute phasor
                    float2 phasor = make_float2(0, 0);
                    float phase = (_ulvmwn[x] * _wavenumbers[chan]) - _phase_offset[x];
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
            int time_offset = chunk * CHUNKSIZE;
            visibilities[bl][time+time_offset][chan][0] = dataXX;
            visibilities[bl][time+time_offset][chan][1] = dataXY;
            visibilities[bl][time+time_offset][chan][2] = dataYX;
            visibilities[bl][time+time_offset][chan][3] = dataYY;
        }
    }
}
}
