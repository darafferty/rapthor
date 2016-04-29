#include "math.cl"
#include "Types.cl"

#define NR_THREADS ((SUBGRIDSIZE % 16 == 0) ? 256 : 64)

/*
	Kernel
*/
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
#if 0
	int s = get_group_id(0);
    int tid = get_local_id(0);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata
	const Metadata m = metadata[s];
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

    // Shared data
    __local float4 _pix[NR_POLARIZATIONS / 2][NR_THREADS];
	__local float4 _lmn_phaseoffset[NR_THREADS];

    for (int i = tid; i < nr_timesteps * nr_channels; i += NR_THREADS) {
		int time = i / nr_channels;
		int chan = i % nr_channels;

        float8 vis = (float8) (0, 0, 0, 0, 0, 0, 0 ,0);
        float4 _uvw;
		float wavenumber;

		if (time < nr_timesteps) {
            UVW a = uvw[offset + time];
            _uvw = (float4) (a.u, a.v, a.w, 0);
			wavenumber = wavenumbers[chan];
		}

		for (int j = tid; j < SUBGRIDSIZE * SUBGRIDSIZE; j += NR_THREADS) {
			int y = j / SUBGRIDSIZE;
			int x = j % SUBGRIDSIZE;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (y < SUBGRIDSIZE) {
                // Load aterm for station1
                float2 aXX1 = aterm[station1][aterm_index][0][y][x];
                float2 aXY1 = aterm[station1][aterm_index][1][y][x];
                float2 aYX1 = aterm[station1][aterm_index][2][y][x];
                float2 aYY1 = aterm[station1][aterm_index][3][y][x];

                // Load aterm for station2
                float2 aXX2 = conj(aterm[station2][aterm_index][0][y][x]);
                float2 aXY2 = conj(aterm[station2][aterm_index][1][y][x]);
                float2 aYX2 = conj(aterm[station2][aterm_index][2][y][x]);
                float2 aYY2 = conj(aterm[station2][aterm_index][3][y][x]);

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

                _pix[0][tid] = (float4) (pixXX.x, pixXX.y, pixXY.x, pixXY.y);
                _pix[1][tid] = (float4) (pixYX.x, pixYX.y, pixYY.x, pixYY.y);

                float l = -(x - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
                float m =  (y - (SUBGRIDSIZE / 2)) * (float) IMAGESIZE / SUBGRIDSIZE;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
                float phase_offset = u_offset * l + v_offset * m + w_offset * n;
                _lmn_phaseoffset[tid] = (float4) (l, m, n, phase_offset);
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
					float  l = _lmn_phaseoffset[k].x;
					float  m = _lmn_phaseoffset[k].y;
					float  n = _lmn_phaseoffset[k].z;
					float  phase_offset = _lmn_phaseoffset[k].w;
					float  phase_index = _uvw.x * l + _uvw.y * m + _uvw.z * n;
					float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

					float2 apXX = (float2) (_pix[0][k].x, _pix[0][k].y);
					float2 apXY = (float2) (_pix[0][k].z, _pix[0][k].w);
					float2 apYX = (float2) (_pix[1][k].x, _pix[1][k].y);
					float2 apYY = (float2) (_pix[1][k].z, _pix[1][k].w);

					vis.s0 += apXX.x * phasor.x;
					vis.s0 -= apXX.y * phasor.y;
					vis.s1 += apXX.x * phasor.y;
					vis.s1 += apXX.y * phasor.x;

					vis.s2 += apXY.x * phasor.x;
					vis.s2 -= apXY.y * phasor.y;
					vis.s3 += apXY.x * phasor.y;
					vis.s3 += apXY.y * phasor.x;

					vis.s4 += apYX.x * phasor.x;
					vis.s4 -= apYX.y * phasor.y;
					vis.s5 += apYX.x * phasor.y;
					vis.s5 += apYX.y * phasor.x;

					vis.s6 += apYY.x * phasor.x;
					vis.s6 -= apYY.y * phasor.y;
					vis.s7 += apYY.x * phasor.y;
					vis.s7 += apYY.y * phasor.x;
				}
			}
        }

        if (time < nr_timesteps) {
            visibilities[offset + time][chan][0] = (float2) (vis.s0, vis.s1);
            visibilities[offset + time][chan][1] = (float2) (vis.s2, vis.s3);
            visibilities[offset + time][chan][2] = (float2) (vis.s4, vis.s5);
            visibilities[offset + time][chan][3] = (float2) (vis.s6, vis.s7);
		}
	}
#endif
}
