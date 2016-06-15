#include "math.cl"
#include "Types.cl"

#define NR_THREADS ((SUBGRIDSIZE % 16 == 0) ? 256 : 64)
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

#define NR_CHANNELS_8 8

/*
	Kernel
*/
__kernel void kernel_degridder_1(
    const float w_offset,
    const int nr_channels,
    const int channel_offset,
    __global const UVWType			uvw,
    __global const WavenumberType	wavenumbers,
    __global       VisibilitiesType	visibilities,
    __global const SpheroidalType	spheroidal,
    __global const ATermType		aterm,
    __global const MetadataType		metadata,
    __global const SubGridType		subgrid,
    __local        float4           _pix[NR_POLARIZATIONS/2][NR_THREADS],
    __local        float4           _lmn_phaseoffset[NR_THREADS]
	) {
    int tidx = get_local_id(0);
	int s = get_group_id(0);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata
	const Metadata m = metadata[s];
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

    // Iterate all timesteps
    for (int time = tidx; time < ALIGN(nr_timesteps, NR_THREADS); time += NR_THREADS) {
        float8 vis = (float8) (0, 0, 0, 0, 0, 0, 0 ,0);
        float4 _uvw;
		float  wavenumber;

		if (time < nr_timesteps) {
            UVW a = uvw[time_offset_global + time];
            _uvw = (float4) (a.u, a.v, a.w, 0);
			wavenumber = wavenumbers[channel_offset];
		}

		for (int j = tidx; j < ALIGN(SUBGRIDSIZE * SUBGRIDSIZE, NR_THREADS); j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (y < SUBGRIDSIZE) {
                // Load aterm for station1
                float2 aXX1 = aterm[aterm_index][station1][y][x][0];
                float2 aXY1 = aterm[aterm_index][station1][y][x][1];
                float2 aYX1 = aterm[aterm_index][station1][y][x][2];
                float2 aYY1 = aterm[aterm_index][station1][y][x][3];

                // Load aterm for station2
                float2 aXX2 = conj(aterm[aterm_index][station2][y][x][0]);
                float2 aXY2 = conj(aterm[aterm_index][station2][y][x][1]);
                float2 aYX2 = conj(aterm[aterm_index][station2][y][x][2]);
                float2 aYY2 = conj(aterm[aterm_index][station2][y][x][3]);

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
                    &pixelsXX, &pixelsXY, &pixelsYX, &pixelsYY);

                // Store pixels
                _pix[0][tidx] = (float4) (pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                _pix[1][tidx] = (float4) (pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                float l = (x-(SUBGRIDSIZE / 2)) * IMAGESIZE/SUBGRIDSIZE;
                float m = (y-(SUBGRIDSIZE / 2)) * IMAGESIZE/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                _lmn_phaseoffset[tidx] = (float4) (l, m, n, phase_offset);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

			if (time < nr_timesteps) {
                #if SUBGRIDSIZE * SUBGRIDSIZE % NR_THREADS == 0
				int last_k = NR_THREADS;
                #else
				int first_j = (j / NR_THREADS) * NR_THREADS;
				int last_k =  first_j + NR_THREADS < SUBGRIDSIZE * SUBGRIDSIZE ? NR_THREADS : SUBGRIDSIZE * SUBGRIDSIZE - first_j;
                #endif

				for (int k = 0; k < last_k; k ++) {
                    // Load l,m,n
					float  l = _lmn_phaseoffset[k].x;
					float  m = _lmn_phaseoffset[k].y;
					float  n = _lmn_phaseoffset[k].z;

                    // Load phase offset
					float  phase_offset = _lmn_phaseoffset[k].w;

                    // Compute phase index
					float  phase_index = _uvw.x * l + _uvw.y * m + _uvw.z * n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                    // Load pixels from local memory
					float2 apXX = (float2) (_pix[0][k].x, _pix[0][k].y);
					float2 apXY = (float2) (_pix[0][k].z, _pix[0][k].w);
					float2 apYX = (float2) (_pix[1][k].x, _pix[1][k].y);
					float2 apYY = (float2) (_pix[1][k].z, _pix[1][k].w);

                    // Multiply pixels by phasor
					vis.s0 += phasor.x * apXX.x;
					vis.s1 += phasor.x * apXX.y;
					vis.s0 -= phasor.y * apXX.y;
					vis.s1 += phasor.y * apXX.x;

					vis.s2 += phasor.x * apXY.x;
					vis.s3 += phasor.x * apXY.y;
					vis.s2 -= phasor.y * apXY.y;
					vis.s3 += phasor.y * apXY.x;

					vis.s4 += phasor.x * apYX.x;
					vis.s5 += phasor.x * apYX.y;
					vis.s4 -= phasor.y * apYX.y;
					vis.s5 += phasor.y * apYX.x;

					vis.s6 += phasor.x * apYY.x;
					vis.s7 += phasor.x * apYY.y;
					vis.s6 -= phasor.y * apYY.y;
					vis.s7 += phasor.y * apYY.x;
				}
			}
        }

        // Set visibility value
        const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
        int index = (time_offset_global + time) * nr_channels + channel_offset;
        if (time < nr_timesteps) {
            visibilities[index][0] = (float2) (vis.s0, vis.s1) * scale;
            visibilities[index][1] = (float2) (vis.s2, vis.s3) * scale;
            visibilities[index][2] = (float2) (vis.s4, vis.s5) * scale;
            visibilities[index][3] = (float2) (vis.s6, vis.s7) * scale;
		}
	}
}

__kernel void kernel_degridder_8(
    const float w_offset,
    const int nr_channels,
    const int channel_offset,
    __global const UVWType			uvw,
    __global const WavenumberType	wavenumbers,
    __global       VisibilitiesType	visibilities,
    __global const SpheroidalType	spheroidal,
    __global const ATermType		aterm,
    __global const MetadataType		metadata,
    __global const SubGridType		subgrid,
    __local        float4           _pix[NR_POLARIZATIONS/2][NR_THREADS],
    __local        float4           _lmn_phaseoffset[NR_THREADS]
	) {
    int s = get_group_id(0);
    int tidx = get_local_id(0);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata
	const Metadata m = metadata[s];
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

    // Iterate timesteps and channels
    for (int i = tidx; i < ALIGN(nr_timesteps * NR_CHANNELS_8, NR_THREADS); i += NR_THREADS) {
		int time = i / NR_CHANNELS_8;
		int chan = i % NR_CHANNELS_8;

        float8 vis = (float8) (0, 0, 0, 0, 0, 0, 0 ,0);
        float4 _uvw;
		float wavenumber;

		if (time < nr_timesteps) {
            UVW a = uvw[time_offset_global + time];
            _uvw = (float4) (a.u, a.v, a.w, 0);
			wavenumber = wavenumbers[chan + channel_offset];
		}

		for (int j = tidx; j < ALIGN(SUBGRIDSIZE * SUBGRIDSIZE, NR_THREADS); j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (y < SUBGRIDSIZE) {
                // Load aterm for station1
                float2 aXX1 = aterm[aterm_index][station1][y][x][0];
                float2 aXY1 = aterm[aterm_index][station1][y][x][1];
                float2 aYX1 = aterm[aterm_index][station1][y][x][2];
                float2 aYY1 = aterm[aterm_index][station1][y][x][3];

                // Load aterm for station2
                float2 aXX2 = conj(aterm[aterm_index][station2][y][x][0]);
                float2 aXY2 = conj(aterm[aterm_index][station2][y][x][1]);
                float2 aYX2 = conj(aterm[aterm_index][station2][y][x][2]);
                float2 aYY2 = conj(aterm[aterm_index][station2][y][x][3]);

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
                    &pixelsXX, &pixelsXY, &pixelsYX, &pixelsYY);

                // Store pixels
                _pix[0][tidx] = (float4) (pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                _pix[1][tidx] = (float4) (pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                float l = (x-(SUBGRIDSIZE / 2)) * IMAGESIZE/SUBGRIDSIZE;
                float m = (y-(SUBGRIDSIZE / 2)) * IMAGESIZE/SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                _lmn_phaseoffset[tidx] = (float4) (l, m, n, phase_offset);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

			if (time < nr_timesteps) {
                #if SUBGRIDSIZE * SUBGRIDSIZE % NR_THREADS == 0
				int last_k = NR_THREADS;
                #else
				int first_j = j / NR_THREADS * NR_THREADS;
				int last_k =  first_j + NR_THREADS < SUBGRIDSIZE * SUBGRIDSIZE ? NR_THREADS : SUBGRIDSIZE * SUBGRIDSIZE - first_j;
                #endif

				for (int k = 0; k < last_k; k ++) {
                    // Load l,m,n
					float  l = _lmn_phaseoffset[k].x;
					float  m = _lmn_phaseoffset[k].y;
					float  n = _lmn_phaseoffset[k].z;

                    // Load phase offset
					float  phase_offset = _lmn_phaseoffset[k].w;

                    // Compute phase index
					float  phase_index = _uvw.x * l + _uvw.y * m + _uvw.z * n;

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                    // Load pixels from local memory
					float2 apXX = (float2) (_pix[0][k].x, _pix[0][k].y);
					float2 apXY = (float2) (_pix[0][k].z, _pix[0][k].w);
					float2 apYX = (float2) (_pix[1][k].x, _pix[1][k].y);
					float2 apYY = (float2) (_pix[1][k].z, _pix[1][k].w);

                    // Multiply pixels by phasor
					vis.s0 += phasor.x * apXX.x;
					vis.s1 += phasor.x * apXX.y;
					vis.s0 -= phasor.y * apXX.y;
					vis.s1 += phasor.y * apXX.x;

					vis.s2 += phasor.x * apXY.x;
					vis.s3 += phasor.x * apXY.y;
					vis.s2 -= phasor.y * apXY.y;
					vis.s3 += phasor.y * apXY.x;

					vis.s4 += phasor.x * apYX.x;
					vis.s5 += phasor.x * apYX.y;
					vis.s4 -= phasor.y * apYX.y;
					vis.s5 += phasor.y * apYX.x;

					vis.s6 += phasor.x * apYY.x;
					vis.s7 += phasor.x * apYY.y;
					vis.s6 -= phasor.y * apYY.y;
					vis.s7 += phasor.y * apYY.x;
				}
			}
        }

        // Set visibility value
        const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
        int index = (time_offset_global + time) * nr_channels + (channel_offset + chan);
        if (time < nr_timesteps) {
            visibilities[index][0] = (float2) (vis.s0, vis.s1) * scale;
            visibilities[index][1] = (float2) (vis.s2, vis.s3) * scale;
            visibilities[index][2] = (float2) (vis.s4, vis.s5) * scale;
            visibilities[index][3] = (float2) (vis.s6, vis.s7) * scale;
		}
	}
}

__kernel void kernel_degridder(
    const float w_offset,
    const int nr_channels,
    __global const UVWType			uvw,
    __global const WavenumberType	wavenumbers,
    __global       VisibilitiesType	visibilities,
    __global const SpheroidalType	spheroidal,
    __global const ATermType		aterm,
    __global const MetadataType		metadata,
    __global const SubGridType		subgrid
	) {
    __local float4 _pix[NR_POLARIZATIONS/2][NR_THREADS];
    __local float4 _lmn_phaseoffset[NR_THREADS];

    int channel_offset = 0;

    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_degridder_8(
            w_offset, nr_channels, channel_offset, uvw, wavenumbers,
            visibilities, spheroidal, aterm, metadata, subgrid,
            _pix, _lmn_phaseoffset);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_degridder_1(
            w_offset, nr_channels, channel_offset, uvw, wavenumbers,
            visibilities, spheroidal, aterm, metadata, subgrid,
            _pix, _lmn_phaseoffset);
    }
}

