#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

/*
	Kernel
*/
__global__ void kernel_gridder(
    const float w_offset,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	const VisibilitiesType	__restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,
	SubGridType				__restrict__ subgrid
	) {
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidx + tidy * blockDim.x;
	int blockSize = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Load metadata
	const Metadata &m = metadata[s];
    const int offset = m.baseline_offset + m.baseline_offset;
    const int nr_timesteps = m.nr_timesteps;
	const int aterm_index = m.aterm_index;
	const int station1 = m.baseline.station1;
	const int station2 = m.baseline.station2;
	const int x_coordinate = m.coordinate.x;
	const int y_coordinate = m.coordinate.y;


    // Shared data
    #define NR_TIMESTEPS 16
    // TODO: introduce compile time constant: MAX_NR_TIMESTEPS
    //       and use this constant to initialize shared memory
	__shared__ float2 _visibilities[NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
	__shared__ UVW _uvw[NR_TIMESTEPS];
	__shared__ float _wavenumbers[NR_CHANNELS];

    // Load wavenumbers
    for (int i = tid; i < NR_CHANNELS; i += blockSize)
        _wavenumbers[i] = wavenumbers[i];

	// Load UVW
	for (int time = tid; time < nr_timesteps; time += blockSize)
		_uvw[time] = uvw[offset + time];

	// Load visibilities
	for (int i = tid; i < nr_timesteps * NR_CHANNELS * NR_POLARIZATIONS; i += blockSize)
		_visibilities[0][0][i] = visibilities[s][0][i];

	syncthreads();

    // Compute u and v offset in wavelenghts
    float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) / IMAGESIZE * 2 * M_PI;
    float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) / IMAGESIZE * 2 * M_PI;

	// Iterate all pixels in subgrid
	for (int y = tidy; y < SUBGRIDSIZE; y += blockDim.y) {
		for (int x = tidx; x < SUBGRIDSIZE; x += blockDim.x) {
			// Load visibilities for all channels and polarizations
			float2 uvXX = {0, 0};
			float2 uvXY = {0, 0};
			float2 uvYX = {0, 0};
			float2 uvYY = {0, 0};

			// Compute l,m,n
			float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
			float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
			float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

			// Iterate all timesteps
			for (int time = 0; time < nr_timesteps; time++) {
				 // Load UVW coordinates
				float u = _uvw[time].u;
				float v = _uvw[time].v;
				float w = _uvw[time].w;

				// Compute phase index
				float phase_index = u*l + v*m + w*n;

				// Compute phase offset
				float phase_offset = u_offset*l + v_offset*m + w_offset*n;

				// Compute phasor
                #pragma unroll 16
				for (int chan = 0; chan < NR_CHANNELS; chan++) {
                    float wavenumber = _wavenumbers[chan];
					float phase = (phase_index * wavenumber) - phase_offset;
					float2 phasor = make_float2(cos(phase), sin(phase));

					// Load visibilities from shared memory
					float2 visXX = _visibilities[time][chan][0];
					float2 visXY = _visibilities[time][chan][1];
					float2 visYX = _visibilities[time][chan][2];
					float2 visYY = _visibilities[time][chan][3];

					// Multiply visibility by phasor
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
			float2 aXX1 = aterm[station1][aterm_index][0][y][x];
			float2 aXY1 = aterm[station1][aterm_index][1][y][x];
			float2 aYX1 = aterm[station1][aterm_index][2][y][x];
			float2 aYY1 = aterm[station1][aterm_index][3][y][x];

			// Get aterm for station2
			float2 aXX2 = aterm[station2][aterm_index][0][y][x];
			float2 aXY2 = aterm[station2][aterm_index][1][y][x];
			float2 aYX2 = aterm[station2][aterm_index][2][y][x];
			float2 aYY2 = aterm[station2][aterm_index][3][y][x];

			// Apply aterm
			float2 tXX, tXY, tYX, tYY;
			Matrix2x2mul(tXX, tXY, tYX, tYY, cuConjf(aXX1), cuConjf(aYX1), cuConjf(aXY1), cuConjf(aYY1), uvXX, uvXY, uvYX, uvYY);
			Matrix2x2mul(tXX, tXY, tYX, tYY, tXX, tXY, tYX, tYY, aXX2, aXY2, aYX2, aYY2);

			// Load spheroidal
			float sph = spheroidal[y][x];

			// Compute shifted position in subgrid
			int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
			int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

			// Set subgrid value
            subgrid[s][0][y_dst][x_dst] = tXX * sph;
            subgrid[s][1][y_dst][x_dst] = tXY * sph;
            subgrid[s][2][y_dst][x_dst] = tYX * sph;
            subgrid[s][3][y_dst][x_dst] = tYY * sph;
        }
	}
}
}
