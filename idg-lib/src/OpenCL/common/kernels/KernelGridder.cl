#define BATCH_SIZE GRIDDER_BATCH_SIZE
#define BLOCK_SIZE GRIDDER_BLOCK_SIZE
#define MAX_NR_CHANNELS 8

/*
    Kernel
*/
__kernel
__attribute__((work_group_size_hint(BLOCK_SIZE, 1, 1)))
void kernel_gridder_(
    const int                current_nr_channels,
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_channels,
    const int                channel_offset,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global const float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global       float2*   subgrid,
    __local float8           visibilities_[BATCH_SIZE][MAX_NR_CHANNELS],
    __local float4           uvw_[BATCH_SIZE],
    __local float            wavenumbers_[MAX_NR_CHANNELS])
{
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);;
    int nr_threads = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
            int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
            int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
            int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);
            subgrid[idx_xx] = (float2) (0, 0);
            subgrid[idx_xy] = (float2) (0, 0);
            subgrid[idx_yx] = (float2) (0, 0);
            subgrid[idx_yy] = (float2) (0, 0);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

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
    float4 uvw_offset = (float4) (
        (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI,
        (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI,
        w_step * ((float)m.coordinate.z + 0.5) * 2 * M_PI,
        0);

    // Load wavenumbers
    for (int i = tid; i < current_nr_channels; i += nr_threads) {
        wavenumbers_[i] = wavenumbers[channel_offset + i];
    }

    // Iterate all timesteps
    int current_nr_timesteps = BATCH_SIZE;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < BATCH_SIZE ?
                               nr_timesteps - time_offset_local : BATCH_SIZE;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load UVW
        for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            uvw_[time] = (float4) (a.u, a.v, a.w, 0);
        }

        // Load visibilities
        for (int i = tid; i < current_nr_timesteps * current_nr_channels; i += nr_threads) {
            int time = i / current_nr_channels;
            int chan = i % current_nr_channels;
            if (time < current_nr_timesteps && chan < current_nr_channels) {
                int idx_time = time_offset_global + time_offset_local + time;
                int idx_chan = channel_offset + chan;
                int idx_vis = index_visibility(nr_channels, idx_time, idx_chan);
                float2 a = visibilities[idx_vis + 0];
                float2 b = visibilities[idx_vis + 1];
                float2 c = visibilities[idx_vis + 2];
                float2 d = visibilities[idx_vis + 3];
                visibilities_[0][i] = (float8) (a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Iterate all pixels in subgrid
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Private pixels
            float8 pixels = (float8) (0);

            // Compute l,m,n
            float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float tmp = (l * l) + (m * m);
            float n = tmp / (1.0f + native_sqrt(1.0f - tmp));
            float4 lmn = (float4) (l, m, n, 0);

            // Iterate all timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                // Load UVW coordinates
                float4 t = uvw_[time];

                // Compute phase index
                float phase_index = dot(t, lmn);

                // Compute phase offset
                float phase_offset = dot(uvw_offset, lmn);

                // Accumulate pixels
                for (int chan = 0; chan < current_nr_channels; chan++) {
                    // Compute phasor
                    float wavenumber = wavenumbers_[chan];
                    float phase = phase_offset - (phase_index * wavenumber);
                    float8 phasor_real = native_cos(phase);
                    float val = native_sin(phase);
                    float8 phasor_imag = (float8) (val, -val, val, -val,
                                                   val, -val, val, -val);

                    // Load visibilities from shared memory
                    float8 vis = visibilities_[time][chan];

                    // Multiply visibility by phasor
                    pixels += phasor_real * vis;
                    pixels += shuffle(phasor_imag * vis, (uint8) (1, 0, 3, 2, 5, 4, 7, 6));
                }
            } // end for time

            // Get aterm for station1
            int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x);
            float2 aXX1 = aterm[station1_idx + 0];
            float2 aXY1 = aterm[station1_idx + 1];
            float2 aYX1 = aterm[station1_idx + 2];
            float2 aYY1 = aterm[station1_idx + 3];
            float8 aterm1 = (float8) (aXX1, aYY1, aXX1, aYY1);
            float8 aterm2 = (float8) (aYX1, aXY1, aYX1, aXY1);

            // Get aterm for station2
            int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
            float2 aXX2 = conj(aterm[station2_idx + 0]);
            float2 aXY2 = conj(aterm[station2_idx + 1]);
            float2 aYX2 = conj(aterm[station2_idx + 2]);
            float2 aYY2 = conj(aterm[station2_idx + 3]);
            float8 aterm3 = (float8) (aXX2, aXX2, aYY2, aYY2);
            float8 aterm4 = (float8) (aYX2, aYX2, aXY2, aXY2);

            float8 pixels_aterm;

            // Apply aterm to pixels: P*A1
            // [ uvXX, uvXY;    [ aXX1, aXY1;
            //   uvYX, uvYY ] *   aYX1, aYY1 ]
            pixels_aterm = (float8) (0);
            pixels_aterm += cmul8(pixels, aterm1);
            pixels_aterm += cmul8(pixels, aterm2);

            // Apply aterm to pixels: A2^H*P
            // [ aXX2, aYX1;      [ uvXX, uvXY;
            //   aXY1, aYY2 ]  *    uvYX, uvYY ]
            pixels = pixels_aterm;
            pixels_aterm = (float8) (0);
            pixels_aterm += cmul8(pixels, aterm3);
            pixels_aterm += cmul8(pixels, aterm4);

            // Load spheroidal
            pixels_aterm *= spheroidal[y * subgrid_size + x];

            // Compute shifted position in subgrid
            int x_dst = (x + (subgrid_size/2)) % subgrid_size;
            int y_dst = (y + (subgrid_size/2)) % subgrid_size;

            // Set subgrid value
            int idx_xx = index_subgrid(subgrid_size, s, 0, y_dst, x_dst);
            int idx_xy = index_subgrid(subgrid_size, s, 1, y_dst, x_dst);
            int idx_yx = index_subgrid(subgrid_size, s, 2, y_dst, x_dst);
            int idx_yy = index_subgrid(subgrid_size, s, 3, y_dst, x_dst);
            subgrid[idx_xx] += pixels_aterm.s01;
            subgrid[idx_xy] += pixels_aterm.s23;
            subgrid[idx_yx] += pixels_aterm.s45;
            subgrid[idx_yy] += pixels_aterm.s67;
        } // end for i
    } // end for time_offset_local
} // end kernel_gridder_

#define KERNEL_GRIDDER_TEMPLATE(NR_CHANNELS) \
    for (; (channel_offset + NR_CHANNELS) <= nr_channels; channel_offset += NR_CHANNELS) { \
        kernel_gridder_( \
            NR_CHANNELS, grid_size, subgrid_size, image_size, w_step, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid, \
            visibilities_, uvw_, wavenumbers_); \
    }

__kernel void kernel_gridder(
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_channels,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global const float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global       float2*   subgrid)
{
    __local float8 visibilities_[BATCH_SIZE][MAX_NR_CHANNELS];
    __local float4 uvw_[BATCH_SIZE];
    __local float  wavenumbers_[MAX_NR_CHANNELS];

    int channel_offset = 0;
    for (int i = 8; i > 0; i--) {
        KERNEL_GRIDDER_TEMPLATE(i);
    }
} // end kernel_gridder

