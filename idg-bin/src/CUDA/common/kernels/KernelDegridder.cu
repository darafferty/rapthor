#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define NR_THREADS DEGRIDDER_BATCH_SIZE
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

/*
	Kernel
*/
template<int current_nr_channels> __device__ void kernel_degridder_(
    const float w_offset,
    const int nr_channels,
    const int channel_offset,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	VisibilitiesType	    __restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,
	const SubGridType	    __restrict__ subgrid
    ) {
    int tidx = threadIdx.x;
	int s = blockIdx.x;

    // Load metadata for first subgrid
    const Metadata &m_0 = metadata[0];

    // Load metadata for current subgrid
	const Metadata &m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + (m.time_offset - m_0.time_offset);
    const int nr_timesteps = m.nr_timesteps;
	const int aterm_index = m.aterm_index;
	const int station1 = m.baseline.station1;
	const int station2 = m.baseline.station2;
	const int x_coordinate = m.coordinate.x;
	const int y_coordinate = m.coordinate.y;

	// Compute u and v offset in wavelenghts
    float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) / IMAGESIZE * 2 * M_PI;
    float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) / IMAGESIZE * 2 * M_PI;

    // Shared data
    __shared__ float4 _pix[NR_POLARIZATIONS / 2][NR_THREADS];
	__shared__ float4 _lmn_phaseoffset[NR_THREADS];

    // Iterate all visibilities
    for (int i = tidx; i < ALIGN(nr_timesteps * current_nr_channels, NR_THREADS); i += NR_THREADS) {
        int time = i / current_nr_channels;
        int chan = i % current_nr_channels;

        float2 visXX, visXY, visYX, visYY;
        float  u, v, w;
        float  wavenumber;

        if (time < nr_timesteps) {
            visXX = make_float2(0, 0);
            visXY = make_float2(0, 0);
            visYX = make_float2(0, 0);
            visYY = make_float2(0, 0);

            u = uvw[time_offset_global + time].u;
            v = uvw[time_offset_global + time].v;
            w = uvw[time_offset_global + time].w;

            wavenumber = wavenumbers[chan + channel_offset];
        }

        __syncthreads();

        for (int j = tidx; j < SUBGRIDSIZE * SUBGRIDSIZE; j += NR_THREADS) {
            int y = j / SUBGRIDSIZE;
            int x = j % SUBGRIDSIZE;

            __syncthreads();

            if (y < SUBGRIDSIZE) {
                float2 aXX1 = aterm[aterm_index][station1][y][x][0];
                float2 aXY1 = aterm[aterm_index][station1][y][x][1];
                float2 aYX1 = aterm[aterm_index][station1][y][x][2];
                float2 aYY1 = aterm[aterm_index][station1][y][x][3];

                // Load aterm for station2
                float2 aXX2 = cuConjf(aterm[aterm_index][station2][y][x][0]);
                float2 aXY2 = cuConjf(aterm[aterm_index][station2][y][x][1]);
                float2 aYX2 = cuConjf(aterm[aterm_index][station2][y][x][2]);
                float2 aYY2 = cuConjf(aterm[aterm_index][station2][y][x][3]);

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

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixelsXX, pixelsXY, pixelsYX, pixelsYY);

                // Store pixels
                _pix[0][tidx] = make_float4(pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                _pix[1][tidx] = make_float4(pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                _lmn_phaseoffset[tidx] = make_float4(l, m, n, phase_offset);
            }
            __syncthreads();

            if (time < nr_timesteps) {
                #if SUBGRIDSIZE * SUBGRIDSIZE % NR_THREADS == 0
                int last_k = NR_THREADS;
                #else
                int first_j = (j / NR_THREADS) * NR_THREADS;
                int last_k =  first_j + NR_THREADS < SUBGRIDSIZE * SUBGRIDSIZE ? NR_THREADS : SUBGRIDSIZE * SUBGRIDSIZE - first_j;
                #endif

                for (int k = 0; k < last_k; k++) {
                    // Load l,m,n
                    float l = _lmn_phaseoffset[k].x;
                    float m = _lmn_phaseoffset[k].y;
                    float n = _lmn_phaseoffset[k].z;

                    // Load phase offset
                    float phase_offset = _lmn_phaseoffset[k].w;

                    // Compute phase index
                    float phase_index = u * l + v * m + w * n;

                    // Compute phasor
                    float  phase  = phase_offset - (phase_index * wavenumber);
                    float2 phasor = make_float2(cosf(phase), sinf(phase));

                    // Load pixels from shared memory
                    float2 apXX = make_float2(_pix[0][k].x, _pix[0][k].y);
                    float2 apXY = make_float2(_pix[0][k].z, _pix[0][k].w);
                    float2 apYX = make_float2(_pix[1][k].x, _pix[1][k].y);
                    float2 apYY = make_float2(_pix[1][k].z, _pix[1][k].w);

                    // Multiply pixels by phasor
                    visXX.x += phasor.x * apXX.x;
                    visXX.y += phasor.x * apXX.y;
                    visXX.x -= phasor.y * apXX.y;
                    visXX.y += phasor.y * apXX.x;

                    visXY.x += phasor.x * apXY.x;
                    visXY.y += phasor.x * apXY.y;
                    visXY.x -= phasor.y * apXY.y;
                    visXY.y += phasor.y * apXY.x;

                    visYX.x += phasor.x * apYX.x;
                    visYX.y += phasor.x * apYX.y;
                    visYX.x -= phasor.y * apYX.y;
                    visYX.y += phasor.y * apYX.x;

                    visYY.x += phasor.x * apYY.x;
                    visYY.y += phasor.x * apYY.y;
                    visYY.x -= phasor.y * apYY.y;
                    visYY.y += phasor.y * apYY.x;
                }
            }
        }

        __syncthreads();

        // Set visibility value
        int vis_offset = (time_offset_global + time) * nr_channels + (channel_offset + chan);
        const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
        if (time < nr_timesteps) {
            visibilities[vis_offset][0] = visXX * scale;
            visibilities[vis_offset][1] = visXY * scale;
            visibilities[vis_offset][2] = visYX * scale;
            visibilities[vis_offset][3] = visYY * scale;
        }
    }
}

extern "C" {
__global__ void kernel_degridder(
    const float w_offset,
    const int nr_channels,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	VisibilitiesType	    __restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,
	const SubGridType	    __restrict__ subgrid
	) {
    int channel_offset = 0;
    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_degridder_<8>(
            w_offset, nr_channels, channel_offset, uvw, wavenumbers,
            visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_degridder_<1>(
            w_offset, nr_channels, channel_offset, uvw, wavenumbers,
            visibilities, spheroidal, aterm, metadata, subgrid);
    }
}
}
