#define BATCH_SIZE DEGRIDDER_BATCH_SIZE
#define BLOCK_SIZE DEGRIDDER_BLOCK_SIZE
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))


/*
    Kernel
*/
__kernel
__attribute__((work_group_size_hint(BLOCK_SIZE, 1, 1)))
void kernel_degridder(
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

    // Load metadata for current subgrid
    const Metadata m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int aterm_index = m.aterm_index;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float)m.coordinate.z + 0.5) * 2 * M_PI;

    // Shared data
    __local float8 pixels_[BATCH_SIZE];
    __local float4 lmn_phaseoffset_[BATCH_SIZE];

    // Iterate visibilities
    for (int i = tid; i < ALIGN(nr_timesteps * nr_channels, nr_threads); i += nr_threads) {
        int time = i / nr_channels;
        int chan = i % nr_channels;

        float8 vis;
        float u, v, w;
        float wavenumber;

        if (time < nr_timesteps) {
            vis = (float8) (0, 0, 0, 0, 0, 0, 0 ,0);
            u = uvw[time_offset_global + time].u;
            v = uvw[time_offset_global + time].v;
            w = uvw[time_offset_global + time].w;
            wavenumber = wavenumbers[chan];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Iterate pixels
        const int nr_pixels = subgrid_size * subgrid_size;
        int current_nr_pixels = BATCH_SIZE;
        for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
            current_nr_pixels = nr_pixels - pixel_offset < min(nr_threads, BATCH_SIZE) ?
                                nr_pixels - pixel_offset : min(nr_threads, BATCH_SIZE);

            barrier(CLK_LOCAL_MEM_FENCE);

            // Prepare data
            for (int j = tid; j < current_nr_pixels; j += nr_threads) {
                int y = (pixel_offset + j) / subgrid_size;
                int x = (pixel_offset + j) % subgrid_size;

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
                float spheroidal_ = spheroidal[y * subgrid_size + x];

                // Compute shifted position in subgrid
                int x_src = (x + (subgrid_size/2)) % subgrid_size;
                int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Load uv values
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
                float2 pixelsXX = spheroidal_ * subgrid[idx_xx];
                float2 pixelsXY = spheroidal_ * subgrid[idx_xy];
                float2 pixelsYX = spheroidal_ * subgrid[idx_yx];
                float2 pixelsYY = spheroidal_ * subgrid[idx_yy];

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    &pixelsXX, &pixelsXY, &pixelsYX, &pixelsYY);

                // Store pixels
                pixels_[j] = (float8) (
                    pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y,
                    pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                const float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                const float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                const float tmp = (l * l) + (m * m);
                const float n = tmp / (1.0f + native_sqrt(1.0f - tmp));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                lmn_phaseoffset_[j] = (float4) (l, m, n, phase_offset);
            } // end for j (pixels)

            barrier(CLK_LOCAL_MEM_FENCE);

            // Iterate current batch of pixels
            for (int k = 0; k < current_nr_pixels; k++) {
                // Load l,m,n
                float l = lmn_phaseoffset_[k].x;
                float m = lmn_phaseoffset_[k].y;
                float n = lmn_phaseoffset_[k].z;

                // Load phase offset
                float phase_offset = lmn_phaseoffset_[k].w;

                // Compute phase index
                float phase_index = u * l + v * m + w * n;

                // Compute phasor
                float  phase  = (phase_index * wavenumber) - phase_offset;
                float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                // Load pixels from local memory
                float2 apXX = (float2) (pixels_[k].s0, pixels_[k].s1);
                float2 apXY = (float2) (pixels_[k].s2, pixels_[k].s3);
                float2 apYX = (float2) (pixels_[k].s4, pixels_[k].s5);
                float2 apYY = (float2) (pixels_[k].s6, pixels_[k].s7);

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
            } // end for k (batch)
        } // end for j (pixels)

        // Store visibility
        const float scale = 1.0f / (nr_pixels);
        int idx_time = time_offset_global + time;
        int idx_vis = index_visibility(nr_channels, idx_time, chan);

        if (time < nr_timesteps) {
            visibilities[idx_vis + 0] = (float2) (vis.s0, vis.s1) * scale;
            visibilities[idx_vis + 1] = (float2) (vis.s2, vis.s3) * scale;
            visibilities[idx_vis + 2] = (float2) (vis.s4, vis.s5) * scale;
            visibilities[idx_vis + 3] = (float2) (vis.s6, vis.s7) * scale;
        }
    } // end for i (visibilities)
} // end kernel_degridder
