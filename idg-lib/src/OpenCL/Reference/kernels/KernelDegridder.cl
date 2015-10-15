#include "math.cl"

#include "Types.cl"

#define NR_THREADS 256
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

/*
	Kernel
*/
__kernel void kernel_degridder(
	const float w_offset,
	__global const UVWType			uvw,
	__global const WavenumberType	wavenumbers,
	__global       VisibilitiesType	visibilities,
	__global const SpheroidalType	spheroidal,
	__global const ATermType		aterm,
	__global const MetadataType		metadata,
	__global const SubGridType		subgrid
	) {
	int s = get_global_id(0);

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


    // Shared data
    __local float4 _pix[NR_POLARIZATIONS / 2][NR_THREADS];
	__local float4 _lmn_phaseoffset[NR_THREADS];

 	for (int i = get_local_id(0); i < ALIGN(NR_TIMESTEPS * NR_CHANNELS, NR_THREADS); i += NR_THREADS) {
		int time = i / NR_CHANNELS;
		int chan = i % NR_CHANNELS;

		float2 visXX, visXY, visYX, visYY;
		float  u, v, w;
		float  wavenumber;

		if (time < NR_TIMESTEPS) {
			visXX = (float2) (0, 0);
			visXY = (float2) (0, 0);
			visYX = (float2) (0, 0);
			visYY = (float2) (0, 0);

			u = uvw[s][time].u;
			v = uvw[s][time].v;
			w = uvw[s][time].w;

			wavenumber = wavenumbers[chan];
		}

		for (int j = get_local_id(0); j < SUBGRIDSIZE * SUBGRIDSIZE; j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

            barrier(CLK_LOCAL_MEM_FENCE);

			if (y < SUBGRIDSIZE) {
                // Load aterm for station1
				float2 aXX1 = aterm[station1][time_nr][0][y][x];
				float2 aXY1 = aterm[station1][time_nr][1][y][x];
				float2 aYX1 = aterm[station1][time_nr][2][y][x];
				float2 aYY1 = aterm[station1][time_nr][3][y][x];

				// Load aterm for station2
				float2 aXX2 = aterm[station2][time_nr][0][y][x];
				float2 aXY2 = aterm[station2][time_nr][1][y][x];
				float2 aYX2 = aterm[station2][time_nr][2][y][x];
				float2 aYY2 = aterm[station2][time_nr][3][y][x];

                //TODO: add conj to one of the stations

				// Load spheroidal
				float _spheroidal = spheroidal[y][x];

				// Compute shifted position in subgrid
				int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
				int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

				// Load uv values
				float2 pixelsXX = _spheroidal * subgrid[s][0][y_src][x_src];
				float2 pixelsXY = _spheroidal * subgrid[s][1][y_src][x_src];
				float2 pixelsYX = _spheroidal * subgrid[s][2][y_src][x_src];
				float2 pixelsYY = _spheroidal * subgrid[s][3][y_src][x_src];

				// Apply aterm to subgrid
				float2 pixXX = pixelsXX * aXX1 + pixelsXY * aYX1 + pixelsXX * aXX2 + pixelsYX * aYX2;
				float2 pixXY = pixelsXX * aXY1 + pixelsXY * aYY1 + pixelsXY * aXX2 + pixelsYY * aYX2;
				float2 pixYX = pixelsYX * aXX1 + pixelsYY * aYX1 + pixelsXX * aXY2 + pixelsYX * aYY2;
				float2 pixYY = pixelsYX * aXY1 + pixelsYY * aYY1 + pixelsXY * aXY2 + pixelsYY * aYY2;

				_pix[0][get_local_id(0)] = (float4) (pixXX.x, pixXX.y, pixXY.x, pixXY.y);
				_pix[1][get_local_id(0)] = (float4) (pixYX.x, pixYX.y, pixYY.x, pixYY.y);

				float l = -(x - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
				float m =  (y - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
				float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
				float phase_offset = u_offset * l + v_offset * m + w_offset * n;
				_lmn_phaseoffset[get_local_id(0)] = (float4) (l, m, n, phase_offset);
			}

            barrier(CLK_LOCAL_MEM_FENCE);

			if (time < NR_TIMESTEPS) {
                #if SUBGRIDSIZE * SUBGRIDSIZE % NR_THREADS == 0
				int last_k = NR_THREADS;
                #else
				int first_j = j / NR_THREADS * NR_THREADS;
				int last_k =  first_j + NR_THREADS < SUBGRIDSIZE * SUBGRIDSIZE ? NR_THREADS : SUBGRIDSIZE * SUBGRIDSIZE - first_j;
                #endif

				for (int k = 0; k < last_k; k ++) {
					float  l = _lmn_phaseoffset[k].x;
					float  m = _lmn_phaseoffset[k].y;
					float  n = _lmn_phaseoffset[k].z;
					float  phase_offset = _lmn_phaseoffset[k].w;
					float  phase_index = u * l + v * m + w * n;
					float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

					float2 apXX = (float2) (_pix[0][k].x, _pix[0][k].y);
					float2 apXY = (float2) (_pix[0][k].z, _pix[0][k].w);
					float2 apYX = (float2) (_pix[1][k].x, _pix[1][k].y);
					float2 apYY = (float2) (_pix[1][k].z, _pix[1][k].w);

					visXX.x += apXX.x * phasor.x;
					visXX.x -= apXX.y * phasor.y;
					visXX.y += apXX.x * phasor.y;
					visXX.y += apXX.y * phasor.x;

					visXY.x += apXY.x * phasor.x;
					visXY.x -= apXY.y * phasor.y;
					visXY.y += apXY.x * phasor.y;
					visXY.y += apXY.y * phasor.x;

					visYX.x += apYX.x * phasor.x;
					visYX.x -= apYX.y * phasor.y;
					visYX.y += apYX.x * phasor.y;
					visYX.y += apYX.y * phasor.x;

					visYY.x += apYY.x * phasor.x;
					visYY.x -= apYY.y * phasor.y;
					visYY.y += apYY.x * phasor.y;
					visYY.y += apYY.y * phasor.x;
				}
			}
		}

        if (time < NR_TIMESTEPS) {
            visibilities[s][time][chan][0] = visXX;
            visibilities[s][time][chan][1] = visXY;
            visibilities[s][time][chan][2] = visYX;
            visibilities[s][time][chan][3] = visYY;
		}
	}
}
