#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

#define NR_THREADS 256
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

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

    // Shared memory
    __shared__ float2 _pix[NR_POLARIZATIONS][NR_THREADS];

    // Map every visibility to one thread
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

        // Iterate all pixels in subgrid
		for (int j = threadIdx.x; j < SUBGRIDSIZE * SUBGRIDSIZE; j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

			__syncthreads();

            // Preprocess pixels and store in shared memory
			if (y < SUBGRIDSIZE) {
                // Load aterm for station1
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

				// Load pixels and apply spheroidal
				float2 pixelsXX = _spheroidal * subgrid[s][0][y_src][x_src];
				float2 pixelsXY = _spheroidal * subgrid[s][1][y_src][x_src];
				float2 pixelsYX = _spheroidal * subgrid[s][2][y_src][x_src];
				float2 pixelsYY = _spheroidal * subgrid[s][3][y_src][x_src];

				// Apply aterm to pixel
				float2 pixXX = pixelsXX * aXX1 + pixelsXY * aYX1 + pixelsXX * aXX2 + pixelsYX * aYX2;
				float2 pixXY = pixelsXX * aXY1 + pixelsXY * aYY1 + pixelsXY * aXX2 + pixelsYY * aYX2;
				float2 pixYX = pixelsYX * aXX1 + pixelsYY * aYX1 + pixelsXX * aXY2 + pixelsYX * aYY2;
				float2 pixYY = pixelsYX * aXY1 + pixelsYY * aYY1 + pixelsXY * aXY2 + pixelsYY * aYY2;

                // Store pixels in shared memory
                _pix[0][threadIdx.x] = pixXX;
                _pix[1][threadIdx.x] = pixXY;
                _pix[2][threadIdx.x] = pixYX;
                _pix[3][threadIdx.x] = pixYY;
			}

			__syncthreads();

            // Iterate all pixels in subgrid
			if (time < NR_TIMESTEPS) {
                #if SUBGRIDSIZE * SUBGRIDSIZE % NR_THREADS == 0
				int last_k = NR_THREADS;
                #else
				int first_j = j / NR_THREADS * NR_THREADS;
				int last_k =  first_j + NR_THREADS < SUBGRIDSIZE * SUBGRIDSIZE ? NR_THREADS : SUBGRIDSIZE * SUBGRIDSIZE - first_j;
                #endif

                // Sum pixel values
				for (int k = 0; k < last_k; k ++) {
                    // Compute l,m,n
				    float l = -(x - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
				    float m =  (y - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
				    float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                    // Compute phase offset
				    float phase_offset = u_offset * l + v_offset * m + w_offset * n;

                    // Compute phasor
					float  phase_index = u * l + v * m + w * n;
					float  phase  = (phase_index * wavenumber) - phase_offset;
					float2 phasor = make_float2(cosf(phase), sinf(phase));

                    // Load pixels
					float2 apXX = _pix[0][k];
					float2 apXY = _pix[1][k];
					float2 apYX = _pix[2][k];
					float2 apYY = _pix[3][k];

                    // Update visibilities
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

        // Store visibilities
		if (time < NR_TIMESTEPS) {
            visibilities[s][time][chan][0] = visXX;
            visibilities[s][time][chan][1] = visXY;
            visibilities[s][time][chan][2] = visYX;
            visibilities[s][time][chan][3] = visYY;
		}
	}
}
}
