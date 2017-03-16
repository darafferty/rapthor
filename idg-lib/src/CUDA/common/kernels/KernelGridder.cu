#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define BATCH_SIZE GRIDDER_BATCH_SIZE

/*
	Kernel
*/
template<int current_nr_channels>
__device__ void kernel_gridder_(
    const int gridsize,
    const float imagesize,
    const float w_offset,
    const int nr_channels,
    const int channel_offset,
    const int nr_stations,
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

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blockSize) {
            subgrid[s][0][0][i] = make_float2(0, 0);
            subgrid[s][1][0][i] = make_float2(0, 0);
            subgrid[s][2][0][i] = make_float2(0, 0);
            subgrid[s][3][0][i] = make_float2(0, 0);
        }
    }

    __syncthreads();

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

    // Shared data
	__shared__ float4 _visibilities[2][BATCH_SIZE][current_nr_channels];
	__shared__ float4 _uvw[BATCH_SIZE];
	__shared__ float _wavenumbers[current_nr_channels];

    // Load wavenumbers
    for (int i = tid; i < current_nr_channels; i += blockSize) {
        _wavenumbers[i] = wavenumbers[channel_offset + i];
    }

    // Iterate all timesteps
    int current_nr_timesteps = BATCH_SIZE;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < BATCH_SIZE ?
                               nr_timesteps - time_offset_local : BATCH_SIZE;

        __syncthreads();

	    // Load UVW
	    for (int time = tid; time < current_nr_timesteps; time += blockSize) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            _uvw[time] = make_float4(a.u, a.v, a.w, 0);
        }

	    // Load visibilities
	    for (int i = tid; i < current_nr_timesteps * current_nr_channels; i += blockSize) {
            int time = i / current_nr_channels;
            int chan = i % current_nr_channels;
            int index = (time_offset_global + time_offset_local + time) * nr_channels + (channel_offset + chan);
            float2 a = visibilities[index][0];
            float2 b = visibilities[index][1];
            float2 c = visibilities[index][2];
            float2 d = visibilities[index][3];
            _visibilities[0][time][chan] = make_float4(a.x, a.y, b.x, b.y);
            _visibilities[1][time][chan] = make_float4(c.x, c.y, d.x, d.y);
        }

	    __syncthreads();

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2 - gridsize/2) / imagesize * 2 * M_PI;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2 - gridsize/2) / imagesize * 2 * M_PI;

	    // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blockSize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Private pixels
            float2 uvXX = make_float2(0, 0);
            float2 uvXY = make_float2(0, 0);
            float2 uvYX = make_float2(0, 0);
            float2 uvYY = make_float2(0, 0);

            // Compute l,m,n
            float l = (x+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float m = (y+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

            // Iterate all timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                // Load UVW coordinates
                float u = _uvw[time].x;
                float v = _uvw[time].y;
                float w = _uvw[time].z;

                // Compute phase index
                float phase_index = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                #pragma unroll current_nr_channels
                for (int chan = 0; chan < current_nr_channels; chan++) {
                    float wavenumber = _wavenumbers[chan];
                    float phase = phase_offset - (phase_index * wavenumber);
                    float2 phasor = make_float2(cos(phase), sin(phase));

                    // Load visibilities from shared memory
                    float4 a = _visibilities[0][time][chan];
                    float4 b = _visibilities[1][time][chan];
                    float2 visXX = make_float2(a.x, a.y);
                    float2 visXY = make_float2(a.z, a.w);
                    float2 visYX = make_float2(b.x, b.y);
                    float2 visYY = make_float2(b.z, b.w);

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
            } // end for time

            // Get aterm for station1
            float2 aXX1 = aterm[aterm_index * nr_stations + station1][y][x][0];
            float2 aXY1 = aterm[aterm_index * nr_stations + station1][y][x][1];
            float2 aYX1 = aterm[aterm_index * nr_stations + station1][y][x][2];
            float2 aYY1 = aterm[aterm_index * nr_stations + station1][y][x][3];

            // Get aterm for station2
            float2 aXX2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][0]);
            float2 aXY2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][1]);
            float2 aYX2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][2]);
            float2 aYY2 = cuConjf(aterm[aterm_index * nr_stations + station2][y][x][3]);

            // Apply aterm
            apply_aterm(
                aXX1, aXY1, aYX1, aYY1,
                aXX2, aXY2, aYX2, aYY2,
                uvXX, uvXY, uvYX, uvYY);

            // Load spheroidal
            float sph = spheroidal[y][x];

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Set subgrid value
            subgrid[s][0][y_dst][x_dst] += uvXX * sph;
            subgrid[s][1][y_dst][x_dst] += uvXY * sph;
            subgrid[s][2][y_dst][x_dst] += uvYX * sph;
            subgrid[s][3][y_dst][x_dst] += uvYY * sph;
	    } // end for i
    } // end for time_offset_local
}

extern "C" {
__global__ void kernel_gridder(
    const int gridsize,
    const float imagesize,
    const float w_offset,
    const int nr_channels,
    const int nr_stations,
	const UVWType			__restrict__ uvw,
	const WavenumberType	__restrict__ wavenumbers,
	const VisibilitiesType	__restrict__ visibilities,
	const SpheroidalType	__restrict__ spheroidal,
	const ATermType			__restrict__ aterm,
	const MetadataType		__restrict__ metadata,
	SubGridType				__restrict__ subgrid
	) {
    int channel_offset = 0;
    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_gridder_<8>(
            gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; (channel_offset + 4) <= nr_channels; channel_offset += 4) {
        kernel_gridder_<4>(
            gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; (channel_offset + 2) <= nr_channels; channel_offset += 2) {
        kernel_gridder_<2>(
            gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_gridder_<1>(
            gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }
}
}
