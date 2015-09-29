#include "math.cl"

#include "Types.cl"


/*
	Kernel
*/
__kernel void kernel_degridder(
	const int bl_offset,
	__global const SubGridType	  subgrid,
	__global const UVWType		  uvw,
	__global const WavenumberType wavenumbers,
	__global const ATermType	  aterm,
	__global const BaselineType	  baselines,
	__global const SpheroidalType spheroidal,
	__global VisibilitiesType	  visibilities
	) {
    int tidx = get_local_id(0);
    int tid = tidx;
    int bl = get_global_id(0) / get_local_size(0);
	 
     // Shared data
    __local UVW _uvw[CHUNKSIZE];
    __local float _wavenumbers[NR_CHANNELS];
    __local fcomplex _uv[NR_POLARIZATIONS][SUBGRIDSIZE];
    __local float _ulvmwn[SUBGRIDSIZE];
    __local float _phase_offset[SUBGRIDSIZE];
    
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
        barrier(CLK_LOCAL_MEM_FENCE); 

        // Compute offset for current chunk
	    UVW uvw_first = _uvw[0];
        UVW uvw_last = _uvw[CHUNKSIZE-1];
	    int u = (GRIDSIZE/2) - ((uvw_first.u + uvw_last.u) / 2);
	    int v = (GRIDSIZE/2) - ((uvw_first.v + uvw_last.v) / 2);
	    int w = (GRIDSIZE/2) - ((uvw_first.w + uvw_last.w) / 2);
	    UVW _offset = {u, v, w};
	  
        // Determe work distribution 
        int nr_compute_threads = SUBGRIDSIZE * 2;
        int nr_load_threads = 0;
        while (nr_load_threads <= 0) {
            nr_compute_threads /= 2;
            nr_load_threads = get_local_size(0) - nr_compute_threads;
        }

        for (int i = get_local_id(0); i < CHUNKSIZE * NR_CHANNELS; i += get_local_size(0)) {
            // Determine time and channel to be computed
            int time = (i / NR_CHANNELS);
            int chan = i % NR_CHANNELS;
        
             // Private data points
            fcomplex dataXX = {0, 0};
            fcomplex dataXY = {0, 0};
            fcomplex dataYX = {0, 0};
            fcomplex dataYY = {0, 0};
        
            // Process one row of subgrid at a time
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                if (get_local_id(0) < nr_compute_threads) {
                    // A number of threads start computing phasor values
                    tid = get_local_id(0);

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
                    tid = get_local_size(0) - nr_compute_threads;

                    // Compute shifted y position in subgrid
                    int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                   
                    // Preload subgrid values and apply spheroidal 
                    for (int x = tid; x < SUBGRIDSIZE; x += nr_load_threads) {
                        // Get spheroidal
                        float s = spheroidal[y][x];

                        // Compute shifted x position in subgrid
                        int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                        // Load subrid pixels
                        #if ORDER == ORDER_BL_P_V_U
                        fcomplex uvXX = s * subgrid[bl][chunk][0][y_src][x_src];
                        fcomplex uvXY = s * subgrid[bl][chunk][1][y_src][x_src];
                        fcomplex uvYX = s * subgrid[bl][chunk][2][y_src][x_src];
                        fcomplex uvYY = s * subgrid[bl][chunk][3][y_src][x_src];
                        #elif ORDER == ORDER_BL_V_U_P
                        fcomplex uvXX = s * subgrid[bl][chunk][y_src][x_src][0];
                        fcomplex uvXY = s * subgrid[bl][chunk][y_src][x_src][1];
                        fcomplex uvYX = s * subgrid[bl][chunk][y_src][x_src][2];
                        fcomplex uvYY = s * subgrid[bl][chunk][y_src][x_src][3];
                        #endif
                        
                        // Store pixels in shared memory
                        _uv[0][x] = uvXX;
                        _uv[1][x] = uvXY;
                        _uv[2][x] = uvYX;
                        _uv[3][x] = uvYY;
                    }

                    // Apply aterm to subgrid values
                    for (int x = tid; x < SUBGRIDSIZE; x += nr_load_threads) {
                        // Load uv values from shared memory
                        fcomplex uvXX = _uv[0][x];
                        fcomplex uvXY = _uv[1][x];
                        fcomplex uvYX = _uv[2][x];
                        fcomplex uvYY = _uv[3][x];
                        
                        // Get aterm for station1
                        fcomplex aXX1 = aterm[station1][0][y][x];
                        fcomplex aXY1 = aterm[station1][1][y][x];
                        fcomplex aYX1 = aterm[station1][2][y][x];
                        fcomplex aYY1 = aterm[station1][3][y][x];

                        // Get aterm for station2
                        fcomplex aXX2 = conj(aterm[station2][0][y][x]);
                        fcomplex aXY2 = conj(aterm[station2][1][y][x]);
                        fcomplex aYX2 = conj(aterm[station2][2][y][x]);
                        fcomplex aYY2 = conj(aterm[station2][3][y][x]);
                        
                        // Initialize corected uv values
                        fcomplex _uvXX = {0, 0};
                        fcomplex _uvXY = {0, 0};
                        fcomplex _uvYX = {0, 0};
                        fcomplex _uvYY = {0, 0};
                        
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
                
                barrier(CLK_LOCAL_MEM_FENCE);

                // Every thread iterates all pixels in current row 
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute phasor
                    fcomplex phasor = (float2) (0, 0);
                    float phase = (_ulvmwn[x] * _wavenumbers[chan]) - _phase_offset[x];
                    phasor.x = native_cos(phase);
                    phasor.y = native_sin(phase);
            
                    // Update data points
                    dataXX += cmul(phasor, _uv[0][x]);
                    dataXY += cmul(phasor, _uv[1][x]);
                    dataYX += cmul(phasor, _uv[2][x]);
                    dataYY += cmul(phasor, _uv[3][x]);
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
