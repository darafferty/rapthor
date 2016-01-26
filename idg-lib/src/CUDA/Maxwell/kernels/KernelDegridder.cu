#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

#define MAX_NR_TIMESTEPS 32
#define BLOCKDIM_Y 4

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
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidx + tidy * blockDim.x;
	int blockSize = blockDim.x * blockDim.y;
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
	const float u_offset = (x_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;
	const float v_offset = (y_coordinate + SUBGRIDSIZE/2) / (float) IMAGESIZE;

    // Shared memory
    __shared__ float2 _visibilities[MAX_NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
    __shared__ float2 _pixels[BLOCKDIM_Y][NR_POLARIZATIONS][SUBGRIDSIZE];

    // Set visibilities to zero
    for (int i = tid; i < NR_POLARIZATIONS * MAX_NR_TIMESTEPS * NR_CHANNELS; i += blockSize) {
        _visibilities[0][0][i] = make_float2(0, 0);
    }

    // Iterate all rows of subgrid
    for (int y = tidy; y < SUBGRIDSIZE; y += blockDim.y) {
        __syncthreads();

        // Preprocess pixels and store in shared memory
        for (int x = tidx; x < SUBGRIDSIZE; x += blockDim.x) {
            // Load aterm for station1
            float2 aXX1 = aterm[station1][aterm_index][0][y][x];
            float2 aXY1 = aterm[station1][aterm_index][1][y][x];
            float2 aYX1 = aterm[station1][aterm_index][2][y][x];
            float2 aYY1 = aterm[station1][aterm_index][3][y][x];

            // Load aterm for station2
            float2 aXX2 = aterm[station2][aterm_index][0][y][x];
            float2 aXY2 = aterm[station2][aterm_index][1][y][x];
            float2 aYX2 = aterm[station2][aterm_index][2][y][x];
            float2 aYY2 = aterm[station2][aterm_index][3][y][x];

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

            // Apply aterm
            float2 pixXX, pixXY, pixYX, pixYY;
            pixXX  = pixelsXX * aXX1;
            pixXX += pixelsXY * aYX1;
            pixXX += pixelsXX * aXX2;
            pixXX += pixelsYX * aYX2;
            pixXY  = pixelsXX * aXY1;
            pixXY += pixelsXY * aYY1;
            pixXY += pixelsXY * aXX2;
            pixXY += pixelsYY * aYX2;
            pixYX  = pixelsYX * aXX1;
            pixYX += pixelsYY * aYX1;
            pixYX += pixelsXX * aXY2;
            pixYX += pixelsYX * aYY2;
            pixYY  = pixelsYX * aXY1;
            pixYY += pixelsYY * aYY1;
            pixYY += pixelsXY * aXY2;
            pixYY += pixelsYY * aYY2;

            // Store pixels in shared memory
            _pixels[threadIdx.y][0][x] = pixXX;
            _pixels[threadIdx.y][1][x] = pixXY;
            _pixels[threadIdx.y][2][x] = pixYX;
            _pixels[threadIdx.y][3][x] = pixYY;
        }

        __syncthreads();

        // Map every visibility to one thread
        for (int i = tidx; i < nr_timesteps * NR_CHANNELS; i += blockDim.x) {
	    	int time = i / NR_CHANNELS;
	    	int chan = i % NR_CHANNELS;

            // Private storage for visibilities
            float2 visXX = make_float2(0, 0);
            float2 visXY = make_float2(0, 0);
            float2 visYX = make_float2(0, 0);
            float2 visYY = make_float2(0, 0);

            // Load uvw
            float u = uvw[offset + time].u;
            float v = uvw[offset + time].v;
            float w = uvw[offset + time].w;

            // Load wavenumber
            float wavenumber = wavenumbers[chan];

            // Iterate all pixels in row of subgrid
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Compute l,m,n
                float l = -(x - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
                float m =  (y - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                // Compute phase index
                float phase_index = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                float phase  = (phase_index * wavenumber) - phase_offset;
                float2 phasor = make_float2(cosf(phase), sinf(phase));

                // Load pixels
                float2 apXX = _pixels[threadIdx.y][0][x];
                float2 apXY = _pixels[threadIdx.y][1][x];
                float2 apYX = _pixels[threadIdx.y][2][x];
                float2 apYY = _pixels[threadIdx.y][3][x];

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

            atomicAdd(&_visibilities[time][chan][0], visXX);
            atomicAdd(&_visibilities[time][chan][1], visXY);
            atomicAdd(&_visibilities[time][chan][2], visYX);
            atomicAdd(&_visibilities[time][chan][3], visYY);
        }
	}

    __syncthreads();

    // Store visibilities
    for (int i = tid; i < nr_timesteps * NR_CHANNELS * NR_POLARIZATIONS; i += blockSize) {
        visibilities[offset][0][i] = _visibilities[0][0][i];
    }
}
}
