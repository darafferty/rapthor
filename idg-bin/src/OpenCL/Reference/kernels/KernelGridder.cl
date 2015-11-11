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
    int s = get_group_id(0);

    // Shared data
	__local float2 _visibilities[NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
	__local float4 _uvw[NR_TIMESTEPS];
	__local float _wavenumbers[NR_CHANNELS];

    // Load wavenumbers
    for (int i = tid; i < NR_CHANNELS; i+= blocksize) {
        _wavenumbers[tid] = wavenumbers[tid];
    }

    // Load UVW
    for (int time = tid; time < NR_TIMESTEPS; time += blocksize) {
        _uvw[time] = (float4) (uvw[s][time].u, uvw[s][time].v, uvw[s][time].w, 0);
    }

    // Load visibilities
    for (int i = tid; i < NR_TIMESTEPS * NR_CHANNELS * NR_POLARIZATIONS; i += blocksize) {
        _visibilities[0][0][i] = visibilities[s][0][0][i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load metadata
	Metadata m = metadata[s];
	int time_nr = m.time_nr;
	int station1 = m.baseline.station1;
	int station2 = m.baseline.station2;
	int x_coordinate = m.coordinate.x;
	int y_coordinate = m.coordinate.y;

	// Compute u and v offset in wavelenghts
	float u_offset = (x_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;
	float v_offset = (y_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;

    // Iterate all pixels in subgrid
    for (int y = tidy; y < SUBGRIDSIZE; y += get_local_size(1)) {
        for (int x = tidx; x < SUBGRIDSIZE; x += get_local_size(0)) {
            // Private subgrid points
            float2 uvXX = (float2) (0, 0);
            float2 uvXY = (float2) (0, 0);
            float2 uvYX = (float2) (0, 0);
            float2 uvYY = (float2) (0, 0);

            // Compute l,m,n
            float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
            float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

            // Iterate all timesteps
            for (int time = 0; time < NR_TIMESTEPS; time++) {
                 // Load UVW coordinates
                float4 uvw = (float4) _uvw[time];

                // Compute phase index
                float ulvmwn = uvw.x*l + uvw.y*m + uvw.z*n;
                //float ulvmwn = u*l + v*m + w*n;

                // Compute phase offset
				float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                #pragma unroll 4
                for (int chan = 0; chan < NR_CHANNELS; chan++) {
                    float phase = (ulvmwn * _wavenumbers[chan]) - phase_offset;
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                    // Load visibilities from shared memory
                    float2 visXX = _visibilities[time][chan][0];
                    float2 visXY = _visibilities[time][chan][1];
                    float2 visYX = _visibilities[time][chan][2];
                    float2 visYY = _visibilities[time][chan][3];

                    // Multiply visibility by phasor
                    #if 0
                    uvXX += cmul(phasor, visXX);
                    uvXY += cmul(phasor, visXY);
                    uvYX += cmul(phasor, visYX);
                    uvYY += cmul(phasor, visYY);
                    #else
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
                    #endif
                }
            }

            // Get a term for station1
            float2 aXX1 = aterm[station1][time_nr][0][y][x];
            float2 aXY1 = aterm[station1][time_nr][1][y][x];
            float2 aYX1 = aterm[station1][time_nr][2][y][x];
            float2 aYY1 = aterm[station1][time_nr][3][y][x];

            // Get aterm for station2
            float2 aXX2 = aterm[station2][time_nr][0][y][x];
            float2 aXY2 = aterm[station2][time_nr][1][y][x];
            float2 aYX2 = aterm[station2][time_nr][2][y][x];
            float2 aYY2 = aterm[station2][time_nr][3][y][x];

            // Apply aterm TODO: add matrix2x2mul template and conj one of the stations
			float2 auvXX = uvXX * aXX1 + uvXY * aYX1 + uvXX * aXX2 + uvXY * aYX2;
			float2 auvXY = uvXX * aXY1 + uvXY * aYY1 + uvXX * aXY2 + uvXY * aYY2;
			float2 auvYX = uvYX * aXX1 + uvYY * aYX1 + uvYX * aXX2 + uvYY * aYX2;
			float2 auvYY = uvYY * aXY1 + uvYY * aYY1 + uvYY * aXY2 + uvYY * aYY2;

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
