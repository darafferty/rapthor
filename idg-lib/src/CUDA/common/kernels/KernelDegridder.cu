#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define BATCH_SIZE DEGRIDDER_BATCH_SIZE
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))


extern "C" {
__global__ void kernel_degridder(
    const int gridsize,
    const float imagesize,
    const float w_offset,
    const int nr_channels,
    const int nr_stations,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	VisibilitiesType	    __restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,
	SubGridType	            __restrict__ subgrid
	) {
    int tidx       = threadIdx.x;
    int tidy       = threadIdx.y;
    int tid        = tidx + tidy * blockDim.y;
    int nr_threads = blockDim.x * blockDim.y;
	int s          = blockIdx.x;

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
    float u_offset = (x_coordinate + SUBGRIDSIZE/2 - gridsize/2) / imagesize * 2 * M_PI;
    float v_offset = (y_coordinate + SUBGRIDSIZE/2 - gridsize/2) / imagesize * 2 * M_PI;

    // Shared data
    __shared__ float4 _pix[NR_POLARIZATIONS / 2][BATCH_SIZE];
	__shared__ float4 _lmn_phaseoffset[BATCH_SIZE];

    __syncthreads();

    // Prepare pixels
    const int nr_pixels = SUBGRIDSIZE * SUBGRIDSIZE;
    for (int i = tid; i < nr_pixels; i+= nr_threads) {
        int y = i / SUBGRIDSIZE;
        int x = i % SUBGRIDSIZE;

        // Load aterm for station1
        float2 aXX1 = aterm[aterm_index * nr_stations + station1][y][x][0];
        float2 aXY1 = aterm[aterm_index * nr_stations + station1][y][x][1];
        float2 aYX1 = aterm[aterm_index * nr_stations + station1][y][x][2];
        float2 aYY1 = aterm[aterm_index * nr_stations + station1][y][x][3];

        // Load aterm for station2
        float2 aXX2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][0]);
        float2 aXY2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][1]);
        float2 aYX2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][2]);
        float2 aYY2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][3]);

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
        subgrid[s][0][y_src][x_src] = pixelsXX;
        subgrid[s][1][y_src][x_src] = pixelsXY;
        subgrid[s][2][y_src][x_src] = pixelsYX;
        subgrid[s][3][y_src][x_src] = pixelsYY;
    }

    // Iterate all visibilities
    for (int i = tid; i < ALIGN(nr_timesteps * nr_channels, nr_threads); i += nr_threads) {
        int time = i / nr_channels;
        int chan = i % nr_channels;

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

            wavenumber = wavenumbers[chan];
        }

        __syncthreads();

        // Iterate all pixels
        int current_nr_pixels = BATCH_SIZE;
        for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
            current_nr_pixels = nr_pixels - pixel_offset < BATCH_SIZE ?
                                nr_pixels - pixel_offset : BATCH_SIZE;

            __syncthreads();

            // Prepare data
            for (int i = tid; i < current_nr_pixels; i += nr_threads) {
                int y = (pixel_offset + i) / SUBGRIDSIZE;
                int x = (pixel_offset + i) % SUBGRIDSIZE;

                // Compute shifted position in subgrid
                int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Load pixels from device memory
                float2 pixelsXX = subgrid[s][0][y_src][x_src];
                float2 pixelsXY = subgrid[s][1][y_src][x_src];
                float2 pixelsYX = subgrid[s][2][y_src][x_src];
                float2 pixelsYY = subgrid[s][3][y_src][x_src];

                // Store pixels
                _pix[0][i] = make_float4(pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                _pix[1][i] = make_float4(pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                float l = (x+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
                float m = (y+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                _lmn_phaseoffset[i] = make_float4(l, m, n, phase_offset);
            }

             __syncthreads();

            // Iterate batch
            for (int j = 0; j < current_nr_pixels; j++) {
                // Load l,m,n
                float l = _lmn_phaseoffset[j].x;
                float m = _lmn_phaseoffset[j].y;
                float n = _lmn_phaseoffset[j].z;

                // Load phase offset
                float phase_offset = _lmn_phaseoffset[j].w;

                // Compute phase index
                float phase_index = u * l + v * m + w * n;

                // Compute phasor
                float  phase  = (phase_index * wavenumber) - phase_offset;
                float2 phasor = make_float2(cosf(phase), sinf(phase));

                // Load pixels from shared memory
                float2 apXX = make_float2(_pix[0][j].x, _pix[0][j].y);
                float2 apXY = make_float2(_pix[0][j].z, _pix[0][j].w);
                float2 apYX = make_float2(_pix[1][j].x, _pix[1][j].y);
                float2 apYY = make_float2(_pix[1][j].z, _pix[1][j].w);

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
            } // end for j (batch)

        } // end for pixels

        // Store visibility
        const float scale = 1.0f / (nr_pixels);
        int index = (time_offset_global + time) * nr_channels + chan;
        if (time < nr_timesteps) {
            visibilities[index][0] = visXX * scale;
            visibilities[index][1] = visXY * scale;
            visibilities[index][2] = visYX * scale;
            visibilities[index][3] = visYY * scale;
        }
    } // end for i (visibilities)

    __syncthreads();
}
}
