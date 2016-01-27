#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define MAX_NR_TIMESTEPS 32
#define NR_THREADS 256
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

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

    // Load metadata for first subgrid
    const Metadata &m_0 = metadata[0];

    // Load metadata for current subgrid
	const Metadata &m = metadata[s];
    const int offset = (m.baseline_offset - m_0.baseline_offset) + (m.time_offset - m_0.time_offset);
    const int nr_timesteps = m.nr_timesteps;
	const int aterm_index = m.aterm_index;
	const int station1 = m.baseline.station1;
	const int station2 = m.baseline.station2;
	const int x_coordinate = m.coordinate.x;
	const int y_coordinate = m.coordinate.y;

	// Compute u and v offset in wavelenghts
	float u_offset = (x_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;
	float v_offset = (y_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;


    __shared__ float4 _pix[NR_POLARIZATIONS / 2][NR_THREADS];
	__shared__ float4 _lmn_phaseoffset[NR_THREADS];

	for (int i = threadIdx.x; i < ALIGN(nr_timesteps * NR_CHANNELS, NR_THREADS); i += NR_THREADS) {
		int time = i / NR_CHANNELS;
		int chan = i % NR_CHANNELS;

		float2 visXX, visXY, visYX, visYY;
		float  u, v, w;
		float  wavenumber;

		if (time < nr_timesteps) {
			visXX = make_float2(0, 0);
			visXY = make_float2(0, 0);
			visYX = make_float2(0, 0);
			visYY = make_float2(0, 0);

			u = uvw[offset + time].u;
			v = uvw[offset + time].v;
			w = uvw[offset + time].w;

			wavenumber = wavenumbers[chan];
		}

		for (int j = threadIdx.x; j < SUBGRIDSIZE * SUBGRIDSIZE; j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

			__syncthreads();

			if (y < SUBGRIDSIZE) {
				float2 aXX1 = aterm[station1][aterm_index][0][y][x];
				float2 aXY1 = aterm[station1][aterm_index][1][y][x];
				float2 aYX1 = aterm[station1][aterm_index][2][y][x];
				float2 aYY1 = aterm[station1][aterm_index][3][y][x];

				// Load aterm for station2
				float2 aXX2 = cuConjf(aterm[station2][aterm_index][0][y][x]);
				float2 aXY2 = cuConjf(aterm[station2][aterm_index][1][y][x]);
				float2 aYX2 = cuConjf(aterm[station2][aterm_index][2][y][x]);
				float2 aYY2 = cuConjf(aterm[station2][aterm_index][3][y][x]);

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

			if (time < nr_timesteps) {
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

		if (time < nr_timesteps) {
		  visibilities[offset + time][chan][0] = visXX;
		  visibilities[offset + time][chan][1] = visXY;
		  visibilities[offset + time][chan][2] = visYX;
		  visibilities[offset + time][chan][3] = visYY;
		}
	}
}
}
