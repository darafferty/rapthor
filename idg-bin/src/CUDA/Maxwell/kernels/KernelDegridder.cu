#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_degridder(
    const float w_offset,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	VisibilitiesType	    __restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,		
	const SubGridType	    __restrict__ subgrid
    ) {
	int s = blockIdx.x;

	// Load metadata
	const Metadata &m = metadata[s];
	int time_nr = m.time_nr;
	int station1 = m.baseline.station1;
	int station2 = m.baseline.station2;
	int x_coordinate = m.coordinate.x;
	int y_coordinate = m.coordinate.y;

	// Compute u and v offset in wavelenghts
	float u_offset = (x_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;
	float v_offset = (y_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;

#define NR_THREADS 256
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

    __shared__ float4 _pix[NR_POLARIZATIONS / 2][NR_THREADS];
	__shared__ float4 _lmn_phaseoffset[NR_THREADS];

#if 1
	for (int i = threadIdx.x; i < ALIGN(NR_TIMESTEPS * NR_CHANNELS, NR_THREADS); i += NR_THREADS) {
		int time = i / NR_CHANNELS;
		int chan = i % NR_CHANNELS;

		float2 visXX, visXY, visYX, visYY;
		float  u, v, w;
		float  wavenumber;
		
		if (time < NR_TIMESTEPS) {
			visXX = make_float2(0, 0);
			visXY = make_float2(0, 0);
			visYX = make_float2(0, 0);
			visYY = make_float2(0, 0);

			u = uvw[s][time].u;
			v = uvw[s][time].v;
			w = uvw[s][time].w;

			wavenumber = wavenumbers[chan];
		}

		for (int j = threadIdx.x; j < SUBGRIDSIZE * SUBGRIDSIZE; j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

			__syncthreads();

			if (y < SUBGRIDSIZE) {
				float2 aXX1 = aterm[station1][time_nr][0][y][x];
				float2 aXY1 = aterm[station1][time_nr][1][y][x];
				float2 aYX1 = aterm[station1][time_nr][2][y][x];
				float2 aYY1 = aterm[station1][time_nr][3][y][x];

				// Load aterm for station2
				float2 aXX2 = cuConjf(aterm[station2][time_nr][0][y][x]);
				float2 aXY2 = cuConjf(aterm[station2][time_nr][1][y][x]);
				float2 aYX2 = cuConjf(aterm[station2][time_nr][2][y][x]);
				float2 aYY2 = cuConjf(aterm[station2][time_nr][3][y][x]);

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

				_pix[0][threadIdx.x] = make_float4(pixXX.x, pixXX.y, pixXY.x, pixXY.y);
				_pix[1][threadIdx.x] = make_float4(pixYX.x, pixYX.y, pixYY.x, pixYY.y);

				float l = -(x - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
				float m =  (y - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
				float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
				float phase_offset = u_offset * l + v_offset * m + w_offset * n;
				_lmn_phaseoffset[threadIdx.x] = make_float4(l, m, n, phase_offset);
			}

			__syncthreads();

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
					float2 phasor = make_float2(cosf(phase), sinf(phase));

					float2 apXX = make_float2(_pix[0][k].x, _pix[0][k].y);
					float2 apXY = make_float2(_pix[0][k].z, _pix[0][k].w);
					float2 apYX = make_float2(_pix[1][k].x, _pix[1][k].y);
					float2 apYY = make_float2(_pix[1][k].z, _pix[1][k].w);

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
#else
	__shared__ float2 _vis[NR_POLARIZATIONS][NR_TIMESTEPS][NR_CHANNELS];
	__shared__ UVW	  _uvw[NR_TIMESTEPS];
	__shared__ float  _wavenumbers[NR_CHANNELS];

    for (int i = threadIdx.x; i < NR_TIMESTEPS; i += blockDim.x)
	  _uvw[i] = uvw[s][i];

    for (int i = threadIdx.x; i < NR_CHANNELS; i += blockDim.x)
	  _wavenumbers[i] = wavenumbers[i];

    for (int i = threadIdx.x; i < NR_TIMESTEPS * NR_CHANNELS * NR_POLARIZATIONS; i += blockDim.x)
	  _vis[0][0][i] = make_float2(0, 0);

	__syncthreads();

    for (int i = threadIdx.x; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blockDim.x) {
		int x = i % SUBGRIDSIZE;
		int y = i / SUBGRIDSIZE;

		float2 aXX1 = aterm[station1][time_nr][0][y][x];
		float2 aXY1 = aterm[station1][time_nr][1][y][x];
		float2 aYX1 = aterm[station1][time_nr][2][y][x];
		float2 aYY1 = aterm[station1][time_nr][3][y][x];

		// Load aterm for station2
		float2 aXX2 = cuConjf(aterm[station2][time_nr][0][y][x]);
		float2 aXY2 = cuConjf(aterm[station2][time_nr][1][y][x]);
		float2 aYX2 = cuConjf(aterm[station2][time_nr][2][y][x]);
		float2 aYY2 = cuConjf(aterm[station2][time_nr][3][y][x]);

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
		float2 apXX = pixelsXX * aXX1 + pixelsXY * aYX1 + pixelsXX * aXX2 + pixelsYX * aYX2;
		float2 apXY = pixelsXX * aXY1 + pixelsXY * aYY1 + pixelsXY * aXX2 + pixelsYY * aYX2;
		float2 apYX = pixelsYX * aXX1 + pixelsYY * aYX1 + pixelsXX * aXY2 + pixelsYX * aYY2;
		float2 apYY = pixelsYX * aXY1 + pixelsYY * aYY1 + pixelsXY * aXY2 + pixelsYY * aYY2;

		float l = -(x - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
		float m =  (y - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
		float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
		float phase_offset = u_offset * l + v_offset * m + w_offset * n;

		for (int t = 0; t < NR_TIMESTEPS; t ++) {
		    int time = (t + threadIdx.x / NR_CHANNELS) % NR_TIMESTEPS;

			float u = _uvw[time].u;
			float v = _uvw[time].v;
			float w = _uvw[time].w;
			float phase_index  = u * l + v * m + w * n;

			for (int c = 0; c < NR_CHANNELS; c ++) {
				int chan = (c + threadIdx.x) % NR_CHANNELS;
				float  phase  = (phase_index * _wavenumbers[chan]) - phase_offset;
				float2 phasor = make_float2(cosf(phase), sinf(phase));

#if 0
				_vis[0][time][chan] += apXX * phasor;
				_vis[1][time][chan] += apXY * phasor;
				_vis[2][time][chan] += apYX * phasor;
				_vis[3][time][chan] += apYX * phasor;
#else
				float2 pXX = _vis[0][time][chan];
				float2 pXY = _vis[1][time][chan];
				float2 pYX = _vis[2][time][chan];
				float2 pYY = _vis[3][time][chan];

				pXX.x += apXX.x * phasor.x;
				pXX.x -= apXX.y * phasor.y;
				pXX.y += apXX.x * phasor.y;
				pXX.y += apXX.y * phasor.x;

				pXY.x += apXY.x * phasor.x;
				pXY.x -= apXY.y * phasor.y;
				pXY.y += apXY.x * phasor.y;
				pXY.y += apXY.y * phasor.x;

				pYX.x += apYX.x * phasor.x;
				pYX.x -= apYX.y * phasor.y;
				pYX.y += apYX.x * phasor.y;
				pYX.y += apYX.y * phasor.x;

				pYY.x += apYY.x * phasor.x;
				pYY.x -= apYY.y * phasor.y;
				pYY.y += apYY.x * phasor.y;
				pYY.y += apYY.y * phasor.x;

				_vis[0][time][chan] = pXX;
				_vis[1][time][chan] = pXY;
				_vis[2][time][chan] = pYX;
				_vis[3][time][chan] = pYY;
#endif
				__syncthreads();
			}
		}
	}

    for (int i = threadIdx.x; i < 0 && NR_TIMESTEPS * NR_CHANNELS * NR_POLARIZATIONS; i += blockDim.x) {
		int time = i / NR_CHANNELS / NR_POLARIZATIONS;
		int chan = (i / NR_POLARIZATIONS) % NR_CHANNELS;
		int pol = i % NR_POLARIZATIONS;

		visibilities[s][time][chan][pol] = _vis[pol][time][chan];
    }
#endif
}
}
