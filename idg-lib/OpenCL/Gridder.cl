#include "math.cl"

/*
    Derived parameters
*/
#define NR_CHUNKS NR_TIME / CHUNKSIZE
typedef struct { int station1, station2; } Baseline;

/*
    Types
*/
typedef struct { float u, v, w; } UVW;
typedef fcomplex VisibilitiesType[NR_BASELINES][NR_TIME][NR_CHANNELS][NR_POLARIZATIONS];
typedef UVW UVWType[NR_BASELINES][NR_TIME];
typedef float WavenumberType[NR_CHANNELS];
typedef fcomplex ATermType[NR_STATIONS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
typedef Baseline BaselineType[NR_BASELINES];
typedef fcomplex GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];

#if !defined ORDER
#define ORDER ORDER_BL_P_V_U
#endif
#if ORDER == ORDER_BL_V_U_P
typedef fcomplex SubGridType[NR_BASELINES][NR_CHUNKS][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];
#elif ORDER == ORDER_BL_P_V_U
typedef fcomplex SubGridType[NR_BASELINES][NR_CHUNKS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
#endif


/*
	Kernel
*/
__kernel void kernel_gridder(
	const int bl_offset,
	__global const UVWType			uvw,
	__global const WavenumberType	wavenumbers,
	__global const VisibilitiesType	visibilities,
	__global const SpheroidalType	spheroidal,
	__global const ATermType		aterm,
	__global const BaselineType		baselines,		
	__global SubGridType			subgrid
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
	int tid = tidx + tidy * get_local_size(0);;
    int bl = get_global_id(0) + get_global_id(1) * get_global_size(0);

    // Shared data
	__local UVW _uvw[CHUNKSIZE];
	__local float _wavenumbers[NR_CHANNELS];
	__local fcomplex _visibilities[NR_CHANNELS][NR_POLARIZATIONS];
	
    // Load wavenumbers
    if (tid < NR_CHANNELS) {
        _wavenumbers[tid] = wavenumbers[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
	    
        // Iterate all pixels in subgrid
        for (int y = tidy; y < SUBGRIDSIZE; y += get_local_size(1)) {
	        for (int x = tidx; x < SUBGRIDSIZE; x += get_local_size(0)) {
	            // Private subgrid points
	            fcomplex uvXX = (float2) 0;
	            fcomplex uvXY = (float2) 0;
	            fcomplex uvYX = (float2) 0;
	            fcomplex uvYY = (float2) 0;
	        
	            // Compute l,m,n
	            float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
	            float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                // Iterate all timesteps in current chunk
	            for (int time = 0; time < CHUNKSIZE && (chunk * CHUNKSIZE) < NR_TIME; time++) {
            	    // Load visibilities for all channels and polarizations
                	for (int i = tid; i < NR_CHANNELS * NR_POLARIZATIONS; i += get_local_size(0) * get_local_size(1)) {
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
                        #if 0
                        phasor_real[chan] = cos(phase);
                        phasor_imag[chan] = sin(phase);
                        #else
                        float _phasor_real = 0;
                        float _phasor_imag = 0;
                        _phasor_imag = sincos(phase, &_phasor_real);
                        phasor_real[chan] = _phasor_real;
                        phasor_imag[chan] = _phasor_imag;
                        #endif
                    }

		            // Sum updates for all channels
		            for (int chan = 0; chan < NR_CHANNELS; chan++) {
			            // Load visibilities from shared memory
			            fcomplex visXX = _visibilities[chan][0];
			            fcomplex visXY = _visibilities[chan][1];
			            fcomplex visYX = _visibilities[chan][2];
			            fcomplex visYY = _visibilities[chan][3];
			                	
			            // Load phasor
                        fcomplex phasor = {phasor_real[chan], phasor_imag[chan]};
			
			            // Multiply visibility by phasor
			            fcomplex updateXX = cmul(phasor, visXX);
			            fcomplex updateXY = cmul(phasor, visXY);
			            fcomplex updateYX = cmul(phasor, visYX);
			            fcomplex updateYY = cmul(phasor, visYY);

                        // Update subgrid point
                        #if 0
			            uvXX = cadd(uvXX, updateXX);
			            uvXY = cadd(uvXY, updateXY);
			            uvYX = cadd(uvYX, updateYX);
			            uvYY = cadd(uvYY, updateYY);
                        #endif
		            }
	            }

	            // Load spheroidal
	            float s = spheroidal[y][x];
	
	            // Stations for current baseline
                int station1 = baselines[bl+bl_offset].station1;
	            int station2 = baselines[bl+bl_offset].station2;
	
                // Get a term for station1
	            fcomplex aXX1 = conj(aterm[station1][0][y][x]);
	            fcomplex aXY1 = conj(aterm[station1][1][y][x]);
	            fcomplex aYX1 = conj(aterm[station1][2][y][x]);
	            fcomplex aYY1 = conj(aterm[station1][3][y][x]);

	            // Get aterm for station2
	            fcomplex aXX2 = aterm[station2][0][y][x];
	            fcomplex aXY2 = aterm[station2][1][y][x];
	            fcomplex aYX2 = aterm[station2][2][y][x];
	            fcomplex aYY2 = aterm[station2][3][y][x];
	
	            // Apply aterm
	            fcomplex _uvXX = uvXX;
	            fcomplex _uvXY = uvXY;
	            fcomplex _uvYX = uvYX;
	            fcomplex _uvYY = uvYY;
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

                #if 0
	            // Apply spheroidal and update uv grid
                #if ORDER == ORDER_BL_P_V_U
	            subgrid[bl][chunk][0][y][x] = uvXX * s;
	            subgrid[bl][chunk][1][y][x] = uvXY * s;
	            subgrid[bl][chunk][2][y][x] = uvYX * s;
	            subgrid[bl][chunk][3][y][x] = uvYY * s;
	            #elif ORDER_BL_V_U_P
	            subgrid[bl][chunk][y][x][0] = uvXX * s;
	            subgrid[bl][chunk][y][x][1] = uvXY * s;
	            subgrid[bl][chunk][y][x][2] = uvYX * s;
	            subgrid[bl][chunk][y][x][3] = uvYY * s;
	            #endif
                #endif
            }
        }
    }
}
