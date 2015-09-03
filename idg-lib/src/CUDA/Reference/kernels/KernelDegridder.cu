#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_degridder(
    const int jobsize,
    const float w_offset,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	VisibilitiesType	    __restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,		
	const SubGridType	    __restrict__ subgrid
    ) {
    int tidx = threadIdx.x;
    int tid = tidx;
    int s = blockIdx.x + blockIdx.y * gridDim.x;
	 
     // Shared data
    __shared__ UVW _uvw[NR_TIMESTEPS];
    __shared__ float _wavenumbers[NR_CHANNELS];
    __shared__ float2 _uv[NR_POLARIZATIONS][SUBGRIDSIZE];
    __shared__ float _ulvmwn[SUBGRIDSIZE];
    __shared__ float _phase_offset[SUBGRIDSIZE];
    
    // Load wavenumbers
    if (tid < NR_CHANNELS) {
        _wavenumbers[tid] = wavenumbers[tid];
    }
    __syncthreads();
   
    // Iterate all subgrids
    for (; s < jobsize; s++) {
        // Load UVW	
        for (int time = tid; time < NR_TIMESTEPS; time += blockDim.x) {
	        _uvw[tid] = uvw[s][tid]; 
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
  
        // Determe work distribution 
        int nr_compute_threads = SUBGRIDSIZE * 2;
        int nr_load_threads = 0;
        while (nr_load_threads <= 0) {
            nr_compute_threads /= 2;
            nr_load_threads = blockDim.x - nr_compute_threads;
        }

        // Iterate all visibilities
        for (int i = threadIdx.x; i < NR_TIMESTEPS * NR_CHANNELS; i += blockDim.x) {
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
		                _phase_offset[x] = u_offset*l + v_offset*m + w_offset*n;
                    }
                } else {
                    // Remaining threads start preparing uvgrid values
                    tid = blockDim.x - nr_compute_threads;

                    // Compute shifted y position in subgrid
                    int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                   
                    // Preload subgrid values and apply spheroidal 
                    for (int x = tid; x < SUBGRIDSIZE; x += nr_load_threads) {
                        // Get spheroidal
                        float sph = spheroidal[y][x];

                        // Compute shifted x position in subgrid
                        int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                        // Load subrid pixels
                        float2 uvXX = sph * subgrid[s][0][y_src][x_src];
                        float2 uvXY = sph * subgrid[s][1][y_src][x_src];
                        float2 uvYX = sph * subgrid[s][2][y_src][x_src];
                        float2 uvYY = sph * subgrid[s][3][y_src][x_src];
                        
                        // Store pixels in shared memory
                        _uv[0][x] = uvXX;
                        _uv[1][x] = uvXY;
                        _uv[2][x] = uvYX;
                        _uv[3][x] = uvYY;
                    }

                    // Apply aterm to subgrid values
                    for (int x = tid; x < SUBGRIDSIZE; x += nr_load_threads) {
                        // Load uv values from shared memory
                        float2 uvXX = _uv[0][x];
                        float2 uvXY = _uv[1][x];
                        float2 uvYX = _uv[2][x];
                        float2 uvYY = _uv[3][x];
                        
                        // Get aterm for station1
                        float2 aXX1 = aterm[station1][time_nr][0][y][x];
                        float2 aXY1 = aterm[station1][time_nr][1][y][x];
                        float2 aYX1 = aterm[station1][time_nr][2][y][x];
                        float2 aYY1 = aterm[station1][time_nr][3][y][x];

                        // Get aterm for station2
                        float2 aXX2 = cuConjf(aterm[station2][time_nr][0][y][x]);
                        float2 aXY2 = cuConjf(aterm[station2][time_nr][1][y][x]);
                        float2 aYX2 = cuConjf(aterm[station2][time_nr][2][y][x]);
                        float2 aYY2 = cuConjf(aterm[station2][time_nr][3][y][x]);
                        
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
                        _uv[0][x] = _uvXX;
                        _uv[1][x] = _uvXY;
                        _uv[2][x] = _uvYX;
                        _uv[3][x] = _uvYY;
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
                    dataXX += phasor * _uv[0][x];
                    dataXY += phasor * _uv[1][x];
                    dataYX += phasor * _uv[2][x];
                    dataYY += phasor * _uv[3][x];
                }
            }
        
            // Set visibilities
            visibilities[s][time][chan][0] = dataXX;
            visibilities[s][time][chan][1] = dataXY;
            visibilities[s][time][chan][2] = dataYX;
            visibilities[s][time][chan][3] = dataYY;
        }
    }
}
}
