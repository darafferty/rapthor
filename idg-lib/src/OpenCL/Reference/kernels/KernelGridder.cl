#include "math.cl"

#include "Types.cl"


/*
	Kernel
*/
__kernel void kernel_gridder(
	const float w_offset,
	__global const UVWType			uvw,
	__global const WavenumberType	wavenumbers,
	__global const VisibilitiesType	visibilities,
	__global const SpheroidalType	spheroidal,
	__global const ATermType		aterm,
	__global const MetadataType		metadata,		
	__global SubGridType			subgrid
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
	int tid = tidx + tidy * get_local_size(0);;
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_global_id(0);

    // Shared data
	__local fcomplex _visibilities[NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
	__local UVW _uvw[NR_TIMESTEPS];
	__local float _wavenumbers[NR_CHANNELS];
	
    // Load wavenumbers
    for (int i = tid; i < NR_CHANNELS; i+= blocksize) {
        _wavenumbers[tid] = wavenumbers[tid];
    }
    
    // Load UVW
    for (int time = tid; time < NR_TIMESTEPS; time += blocksize) {
        _uvw[time] = uvw[s][time];
    }

    // Load visibilities
    for (int i = tid; i < NR_TIMESTEPS * NR_CHANNELS * NR_POLARIZATIONS; i += blocksize) {
        _visibilities[0][0][i] = visibilities[s][0][0][i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load metadata
	const Metadata *m = &(metadata[s]);
	int time_nr = m->time_nr;
	int station1 = m->baseline.station1;
	int station2 = m->baseline.station2;
	int x_coordinate = m->coordinate.x;
	int y_coordinate = m->coordinate.y;

	// Compute u and v offset in wavelenghts
	float u_offset = (x_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;
	float v_offset = (y_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;
	
    // Iterate all pixels in subgrid
    for (int y = tidy; y < SUBGRIDSIZE; y += get_local_size(1)) {
        for (int x = tidx; x < SUBGRIDSIZE; x += get_local_size(0)) {
            // Private subgrid points
            fcomplex uvXX = (fcomplex) (0, 0);
            fcomplex uvXY = (fcomplex) (0, 0);
            fcomplex uvYX = (fcomplex) (0, 0);
            fcomplex uvYY = (fcomplex) (0, 0);
        
            // Compute l,m,n
            float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
            float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

            // Iterate all timesteps
            for (int time = 0; time < NR_TIMESTEPS; time++) {
                 // Load UVW coordinates
                float u = _uvw[time].u;
                float v = _uvw[time].v;
                float w = _uvw[time].w;
    
                // Compute phase index
                float ulvmwn = u*l + v*m + w*n;

                // Compute phase offset
				float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                                     
                // Compute phasor
                for (int chan = 0; chan < NR_CHANNELS; chan++) {
                    float phase = (ulvmwn * _wavenumbers[chan]) - phase_offset;
                    fcomplex phasor = (fcomplex) (native_cos(phase), native_sin(phase));

                    // Load visibilities from shared memory
                    fcomplex visXX = _visibilities[time][chan][0];
                    fcomplex visXY = _visibilities[time][chan][1];
                    fcomplex visYX = _visibilities[time][chan][2];
                    fcomplex visYY = _visibilities[time][chan][3];
        
                    // Multiply visibility by phasor
                    //uvXX += cmul(phasor, visXX);
                    //uvXY += cmul(phasor, visXY);
                    //uvYX += cmul(phasor, visYX);
                    //uvYY += cmul(phasor, visYY);

					uvXX.x += phasor.x * visXX.x;
					uvXX.y += phasor.x * visXX.y;
					uvXX.x -= phasor.y * visXX.y;
					uvXX.y += phasor.y * visXX.x;

					uvXY.x += phasor.x * visXY.x;
					uvXY.y += phasor.x * visXY.y;
					uvXY.x -= phasor.y * visXY.y;
					uvXY.y += phasor.y * visXY.x;

					uvYX.x += phasor.x * visYX.x;
					uvYX.y += phasor.x * visYX.y;
					uvYX.x -= phasor.y * visYX.y;
					uvYX.y += phasor.y * visYX.x;

					uvYY.x += phasor.x * visYY.x;
					uvYY.y += phasor.x * visYY.y;
					uvYY.x -= phasor.y * visYY.y;
					uvYY.y += phasor.y * visYY.x;
                }
            }

            // Get a term for station1
            fcomplex aXX1 = aterm[station1][time_nr][0][y][x];
            fcomplex aXY1 = aterm[station1][time_nr][1][y][x];
            fcomplex aYX1 = aterm[station1][time_nr][2][y][x];
            fcomplex aYY1 = aterm[station1][time_nr][3][y][x];

            // Get aterm for station2
            fcomplex aXX2 = aterm[station2][time_nr][0][y][x];
            fcomplex aXY2 = aterm[station2][time_nr][1][y][x];
            fcomplex aYX2 = aterm[station2][time_nr][2][y][x];
            fcomplex aYY2 = aterm[station2][time_nr][3][y][x];

            // Apply aterm TODO: add matrix2x2mul template and conj one of the stations
			fcomplex auvXX = uvXX * aXX1 + uvXY * aYX1 + uvXX * aXX2 + uvXY * aYX2;
			fcomplex auvXY = uvXX * aXY1 + uvXY * aYY1 + uvXX * aXY2 + uvXY * aYY2;
			fcomplex auvYX = uvYX * aXX1 + uvYY * aYX1 + uvYX * aXX2 + uvYY * aYX2;
			fcomplex auvYY = uvYY * aXY1 + uvYY * aYY1 + uvYY * aXY2 + uvYY * aYY2;

            // Load spheroidal
            float sph = spheroidal[y][x];

			// Compute shifted position in subgrid
			int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
			int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Apply spheroidal and update uv grid
            subgrid[s][0][y_dst][x_dst] = auvXX * sph;
            subgrid[s][1][y_dst][x_dst] = auvXY * sph;
            subgrid[s][2][y_dst][x_dst] = auvYX * sph;
            subgrid[s][3][y_dst][x_dst] = auvYY * sph;
        }
    }
}
