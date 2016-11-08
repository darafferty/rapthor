#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define BATCH_SIZE DEGRIDDER_BATCH_SIZE
//#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))


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
	const SubGridType	    __restrict__ subgrid
	) {
    int tidx      = threadIdx.x;
    int tidy      = threadIdx.y;
    int tid       = tidx + tidy * blockDim.y;
    int blockSize = blockDim.x * blockDim.y;
	int s         = blockIdx.x;

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

    // Iterate all visibilities
    for (int i = tid; i < nr_timesteps * nr_channels; i += blockSize) {
        int time = i / nr_channels;
        int chan = i % nr_channels;

        float2 visXX, visXY, visYX, visYY;
        float  u, v, w;
        float  wavenumber;

        visXX = make_float2(0, 0);
        visXY = make_float2(0, 0);
        visYX = make_float2(0, 0);
        visYY = make_float2(0, 0);
        
        u = uvw[time_offset_global + time].u;
        v = uvw[time_offset_global + time].v;
        w = uvw[time_offset_global + time].w;
        
        wavenumber = wavenumbers[chan];

        __syncthreads();

        for (int j = 0; j < SUBGRIDSIZE * SUBGRIDSIZE; j++) {
            int y = j / SUBGRIDSIZE;
            int x = j % SUBGRIDSIZE;

            __syncthreads();

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
            //_pix[0][tid] = make_float4(pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
            //_pix[1][tid] = make_float4(pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);
            
            // Compute l,m,n and phase offset
            float l = (x-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float m = (y-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
            float phase_offset = u_offset*l + v_offset*m + w_offset*n;
            //_lmn_phaseoffset[tid] = make_float4(l, m, n, phase_offset);

            __syncthreads();

            //if (time < nr_timesteps) {
                //int last_k = 0;
                //if (SUBGRIDSIZE * SUBGRIDSIZE % blockSize == 0) {
                //    last_k = BATCH_SIZE;
                //} else {
                //    int first_j = (j / BATCH_SIZE) * BATCH_SIZE;
                //    last_k =  first_j + BATCH_SIZE < SUBGRIDSIZE * SUBGRIDSIZE ? BATCH_SIZE : SUBGRIDSIZE * SUBGRIDSIZE - first_j;
                //}

                //for (int k = 0; k < last_k; k++) {
                //    // Load l,m,n
                //    float l = _lmn_phaseoffset[k].x;
                //    float m = _lmn_phaseoffset[k].y;
                //    float n = _lmn_phaseoffset[k].z;

                //    // Load phase offset
                //    float phase_offset = _lmn_phaseoffset[k].w;

            // Compute phase index
            float phase_index = u * l + v * m + w * n;
            
            // Compute phasor
            float  phase  = (phase_index * wavenumber) - phase_offset;
            float2 phasor = make_float2(cosf(phase), sinf(phase));
            
            // Load pixels from shared memory
            //float2 apXX = make_float2(_pix[0][k].x, _pix[0][k].y);
            //float2 apXY = make_float2(_pix[0][k].z, _pix[0][k].w);
            //float2 apYX = make_float2(_pix[1][k].x, _pix[1][k].y);
            //float2 apYY = make_float2(_pix[1][k].z, _pix[1][k].w);
            float2 apXX = pixelsXX;
            float2 apXY = pixelsXY;
            float2 apYX = pixelsYX;
            float2 apYY = pixelsYY;
            
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
                //} // end for k
            //} // end if
        } // end for j (pixels)

        __syncthreads();

        // Set visibility value
        const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
        int index = (time_offset_global + time) * nr_channels + chan;
        if (time < nr_timesteps) {
            visibilities[index][0] = visXX * scale;
            visibilities[index][1] = visXY * scale;
            visibilities[index][2] = visYX * scale;
            visibilities[index][3] = visYY * scale;
        }
    } // end for i (visibilities)
}
}
