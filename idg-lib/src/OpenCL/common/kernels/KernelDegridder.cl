#include "math.cl"
#include "Types.cl"

#define BATCH_SIZE DEGRIDDER_BATCH_SIZE
#define MAX_NR_CHANNELS 8

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))


/*
    Kernel
*/
__kernel void kernel_degridder(
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_channels,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global       float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global const float2*   subgrid)
{
    int s = get_group_id(0);
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid  = tidx + tidy * get_local_size(0);
    int nr_threads = get_local_size(0) * get_local_size(1);

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
    const float w_offset = w_step * m.coordinate.z;
    const float w_offset_in_lambda = w_step * ((float)m.coordinate.z + 0.5);

    // Compute u and v offset in wavelenghts
    float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;

    // Shared data
    __local float4 _pix[NR_POLARIZATIONS/2][BATCH_SIZE];
    __local float4 _lmn_phaseoffset[BATCH_SIZE];

    // Iterate timesteps and channels
    for (int i = tid; i < ALIGN(nr_timesteps * nr_channels, nr_threads); i += nr_threads) {
        int time = i / nr_channels;
        int chan = i % nr_channels;

        float8 vis = (float8) (0, 0, 0, 0, 0, 0, 0 ,0);
        float4 _uvw;
        float wavenumber;

        if (time < nr_timesteps) {
            UVW a = uvw[time_offset_global + time];
            _uvw = (float4) (a.u, a.v, a.w, 0);
            wavenumber = wavenumbers[chan];
        }

        // Prepare pixels
        const int nr_pixels = subgrid_size * subgrid_size;
        for (int j = tid; j < ALIGN(subgrid_size * subgrid_size, nr_threads); j += nr_threads) {
            int y = j / subgrid_size;
            int x = j % subgrid_size;

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (y < subgrid_size) {
                // Load aterm for station1
                int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x);
                float2 aXX1 = aterm[station1_idx + 0];
                float2 aXY1 = aterm[station1_idx + 1];
                float2 aYX1 = aterm[station1_idx + 2];
                float2 aYY1 = aterm[station1_idx + 3];

                // Load aterm for station2
                int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
                float2 aXX2 = conj(aterm[station2_idx + 0]);
                float2 aXY2 = conj(aterm[station2_idx + 1]);
                float2 aYX2 = conj(aterm[station2_idx + 2]);
                float2 aYY2 = conj(aterm[station2_idx + 3]);

                // Load spheroidal
                float _spheroidal = spheroidal[y * subgrid_size + x];

                // Compute shifted position in subgrid
                int x_src = (x + (subgrid_size/2)) % subgrid_size;
                int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Load uv values
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
                float2 pixelsXX = _spheroidal * subgrid[idx_xx];
                float2 pixelsXY = _spheroidal * subgrid[idx_xy];
                float2 pixelsYX = _spheroidal * subgrid[idx_yx];
                float2 pixelsYY = _spheroidal * subgrid[idx_yy];

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    &pixelsXX, &pixelsXY, &pixelsYX, &pixelsYY);

                // Store pixels
                _pix[0][tid] = (float4) (pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                _pix[1][tid] = (float4) (pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                const float l = (x+0.5-(subgrid_size / 2)) * image_size/subgrid_size;
                const float m = (y+0.5-(subgrid_size / 2)) * image_size/subgrid_size;
                const float tmp = (l * l) + (m * m);
                const float n = tmp / (1.0f + native_sqrt(1.0f - tmp));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                _lmn_phaseoffset[tid] = (float4) (l, m, n, phase_offset);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if (time < nr_timesteps) {
                int last_k = nr_threads;
                if (nr_pixels % nr_threads != 0) {
                    int first_j = j / nr_threads * nr_threads;
                    last_k =  first_j + nr_threads < subgrid_size * subgrid_size ? nr_threads : subgrid_size * subgrid_size - first_j;
                }

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
        const float scale = 1.0f / (subgrid_size*subgrid_size);
        int idx_time = time_offset_global + time;
        int idx_xx = index_visibility(nr_channels, idx_time, chan);
        int idx_xy = index_visibility(nr_channels, idx_time, chan);
        int idx_yx = index_visibility(nr_channels, idx_time, chan);
        int idx_yy = index_visibility(nr_channels, idx_time, chan);

        if (time < nr_timesteps) {
            visibilities[idx_xx + 0] = (float2) (vis.s0, vis.s1) * scale;
            visibilities[idx_xy + 1] = (float2) (vis.s2, vis.s3) * scale;
            visibilities[idx_yx + 2] = (float2) (vis.s4, vis.s5) * scale;
            visibilities[idx_yy + 3] = (float2) (vis.s6, vis.s7) * scale;
        }
    }
}
